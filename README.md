# wan2_2_video_service

把 `Wan2.2-Animate-14B` 封装成可远程调用的服务：

- 服务端接收 `prompt + base64 图片`。
- 服务端保存输入到 `results/`（带时间戳和唯一任务 ID）。
- 服务端异步推理（长耗时不会阻塞 HTTP 请求）。
- 服务端输出视频到 `outputs/`（带时间戳和唯一任务 ID）。
- 客户端通过 `SERVER_URL` 访问服务端（可跨机器，内网调用）。


## 目录说明

- `server/animate_service.py`：核心服务、任务队列、推理执行。
- `server/run_server.py`：按 `SERVER_URL` 启动服务。
- `client/submit_animate.py`：客户端提交任务并轮询状态。
- `scripts/download_model.py`：统一下载入口（HF / ModelScope）。
- `scripts/download_huggingface.py`：从 HuggingFace 下载权重。
- `scripts/download_modelscope.py`：从 ModelScope 下载权重。

## 1. 环境准备（uv）

### 1.1 安装 uv

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 安装项目依赖

在项目根目录执行：

```bash
uv sync
uv sync --extra dev linux-gpu
```


## 2. 配置 .env

参考 `.env.example`，至少设置这些变量：

```env
HT_HOME=./modelsweights
SERVER_URL=http://0.0.0.0:1111
WAN_ANIMATE_CKPT_DIR=./modelsweights/Wan2.2-Animate-14B
CUDA_DEVICE_ID=0
RESULTS_ROOT=./results
OUTPUTS_ROOT=./outputs
ANIMATE_SAMPLE_SOLVER=dpm++
HF_MIRROR=https://hf-mirror.com
```

说明：

- 服务端机器把 `SERVER_URL` 设为 `http://0.0.0.0:端口`，用于内网开放监听。
- 客户端机器把 `SERVER_URL` 设为 `http://<服务端内网IP>:端口`，用于访问服务端。

## 3. 下载权重

默认下载到 `HT_HOME`（例如 `./modelsweights`）。

### 3.1 下载 Wan2.2-Animate-14B（服务推理必须）

```bash
uv run python scripts/download_model.py --source huggingface --model Wan2.2-Animate-14B
```

### 3.2 下载 Wan2.2-I2V-A14B（按需）

```bash
uv run python scripts/download_model.py --source huggingface --model Wan2.2-I2V-A14B
```

也可用 ModelScope：

```bash
uv run python scripts/download_model.py --source modelscope --model Wan2.2-Animate-14B
uv run python scripts/download_model.py --source modelscope --model Wan2.2-I2V-A14B
```

## 4. 启动服务端

在服务端机器执行：

```bash
uv run python server/run_server.py
```

健康检查：

```bash
curl http://127.0.0.1:1111/healthz
```

注意：

- 当前服务要求 CUDA GPU；没有 CUDA 时任务会失败并返回明确错误。
- 推理是异步队列执行，请通过任务状态接口轮询，不要等单个长连接返回最终视频。

## 5. 启动客户端并调用

在客户端机器执行：

```bash
uv run python client/submit_animate.py \
  --image /path/to/input.png \
  --prompt "A person waves to the camera"
```

客户端会：

- 把图片转为 base64。
- 调用 `POST /v1/animate/jobs` 创建任务。
- 轮询 `GET /v1/animate/jobs/{job_id}` 直到成功或失败。

## 6. API 输入输出

### 6.1 创建任务

`POST /v1/animate/jobs`

请求 JSON：

```json
{
  "prompt": "A person waves to the camera",
  "image_base64": "<base64-image>",
  "sample_steps": 20,
  "clip_len": 77,
  "fps": 30,
  "seed": 12345,
  "offload_model": true
}
```

响应 JSON（示例）：

```json
{
  "job_id": "20260413_120636_01bf98b6_2ccfb89f",
  "status": "queued",
  "status_url": "http://<host>:1111/v1/animate/jobs/20260413_120636_01bf98b6_2ccfb89f"
}
```

### 6.2 查询任务

`GET /v1/animate/jobs/{job_id}`

状态：

- `queued`
- `running`
- `succeeded`
- `failed`

成功时返回：

- `output_video_path`：服务端本地文件路径。
- `output_video_url`：可通过 HTTP 访问的视频地址（`/outputs/...`）。

失败时返回：

- `error`：错误信息。

## 7. 输入输出文件落盘规则

每个任务都会有独立目录和唯一文件名（时间戳 + hash + job id）：

- 输入图：`results/job_<job_id>/input_<job_id>.png`
- 推理输入：`results/job_<job_id>/src_ref.png`、`src_pose.mp4`、`src_face.mp4`
- 输出视频：`outputs/wan_animate_<job_id>.mp4`

## 8. 一组最小可运行命令

服务端：

```bash
uv sync
uv run python scripts/download_model.py --source huggingface --model Wan2.2-Animate-14B
uv run python server/run_server.py
```

客户端：

```bash
uv run python client/submit_animate.py --image ./inputs/demo.png --prompt "A person dancing"
```
