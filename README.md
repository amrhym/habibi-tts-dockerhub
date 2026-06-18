# Habibi-TTS Ready Container

Ready container for validating SWivid/Habibi-TTS on NVIDIA B200/Blackwell.

The image exposes FastAPI on port `8000`:

- `GET /health`
- `GET /ready`
- `POST /infer` returns `audio/wav`
- `POST /infer_json` returns base64 WAV

## Docker Hub Automated Build

Create a private Docker Hub repo, for example:

```text
amrhym/habibi-tts-b200
```

Connect Docker Hub to the GitHub repo that contains these files.

Build settings:

```text
Source: GitHub
Branch: main
Dockerfile location: /Dockerfile
Build context: /
Tag: latest
Architecture: linux/amd64
Visibility: Private
```

Default base image:

```text
pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
```

For an on-prem NVIDIA Enterprise build, override build arg `BASE_IMAGE` with:

```text
nvcr.io/nvidia/pytorch:25.03-py3
```

Do not use public visibility unless the Habibi model license and customer
policy explicitly allow redistribution of the bundled weights.

## Manual Build

Use a Linux AMD64 machine with Docker:

```bash
docker build -t amrhym/habibi-tts-b200:latest .
docker push amrhym/habibi-tts-b200:latest
```

Optional NGC base:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 \
  -t amrhym/habibi-tts-b200:ngc-25.03 .
```

## RunPod Test

Create a RunPod pod using:

```text
Image: amrhym/habibi-tts-b200:latest
GPU: B200, H200, H100, or L40S
Expose HTTP port: 8000
Container disk: 80 GB minimum
Volume: optional
```

Runtime command and args should be empty because the image already has:

```text
CMD ["python3", "/app/server.py"]
```

Test:

```bash
BASE_URL=https://<pod-id>-8000.proxy.runpod.net ./smoke_test.sh
```

or:

```bash
curl -fsS https://<pod-id>-8000.proxy.runpod.net/health

curl -fsS \
  -H 'Content-Type: application/json' \
  -X POST https://<pod-id>-8000.proxy.runpod.net/infer \
  -d '{"text":"هلا والله، كيف أقدر أخدمك اليوم؟","dialect":"SAU","model":"Unified"}' \
  -o habibi-test.wav
```

## Run:ai Settings

```text
Protocol: HTTP
Container port: 8000
Command: empty
Arguments: empty
```

Do not use `vllm`; Habibi-TTS is served by the Python FastAPI wrapper.
