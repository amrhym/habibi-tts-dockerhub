import base64
import importlib.metadata
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


MODEL_DIR = Path(os.getenv("HABIBI_MODEL_DIR", "/models/Habibi-TTS"))
OUTPUT_DIR = Path(os.getenv("HABIBI_OUTPUT_DIR", "/tmp/habibi-tts"))
DEFAULT_MODEL = os.getenv("HABIBI_DEFAULT_MODEL", "Unified")
DEFAULT_DIALECT = os.getenv("HABIBI_DEFAULT_DIALECT", "SAU")
DEFAULT_REF_AUDIO = os.getenv("HABIBI_DEFAULT_REF_AUDIO", "assets/Najdi.wav")
DEFAULT_REF_TEXT = os.getenv(
    "HABIBI_DEFAULT_REF_TEXT",
    "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه.",
)

ALLOWED_DIALECTS = {
    "UNK",
    "MSA",
    "SAU",
    "UAE",
    "ALG",
    "IRQ",
    "EGY",
    "MAR",
    "OMN",
    "TUN",
    "LEV",
    "SDN",
    "LBY",
}


class InferRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    dialect: str = DEFAULT_DIALECT
    model: Literal["Unified", "Specialized"] = DEFAULT_MODEL  # type: ignore[assignment]
    ref_audio: str | None = None
    ref_text: str | None = None
    output_file: str | None = None
    remove_silence: bool = False
    nfe_step: int | None = Field(default=None, ge=1, le=128)
    speed: float | None = Field(default=None, gt=0.2, le=3.0)
    timeout_seconds: int = Field(default=240, ge=30, le=1800)


app = FastAPI(title="Habibi-TTS Runtime", version="1.0.0")


def _model_paths(model: str, dialect: str) -> tuple[Path, Path]:
    if model == "Unified":
        base = MODEL_DIR / "Unified"
        ckpt = base / "model_200000.safetensors"
        vocab = base / "vocab.txt"
    elif model == "Specialized" and dialect == "SAU":
        base = MODEL_DIR / "Specialized" / "SAU"
        ckpt = base / "model_200000.safetensors"
        vocab = base / "vocab.txt"
    else:
        raise HTTPException(
            status_code=400,
            detail="This image includes Unified and Specialized/SAU only.",
        )

    missing = [str(path) for path in (ckpt, vocab) if not path.exists()]
    if missing:
        raise HTTPException(status_code=500, detail={"missing_model_files": missing})
    return ckpt, vocab


def _base_status() -> dict:
    model_files = {
        "unified_ckpt": (MODEL_DIR / "Unified" / "model_200000.safetensors").exists(),
        "unified_vocab": (MODEL_DIR / "Unified" / "vocab.txt").exists(),
        "sau_ckpt": (MODEL_DIR / "Specialized" / "SAU" / "model_200000.safetensors").exists(),
        "sau_vocab": (MODEL_DIR / "Specialized" / "SAU" / "vocab.txt").exists(),
    }
    return {
        "ok": all(model_files.values()),
        "model_dir": str(MODEL_DIR),
        "model_files": model_files,
        "habibi_tts": importlib.metadata.version("habibi-tts"),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/health")
def health() -> dict:
    return _base_status()


@app.get("/ready")
def ready() -> dict:
    status = _base_status()
    if not status["ok"]:
        raise HTTPException(status_code=503, detail=status)
    return status


def _run_inference(req: InferRequest) -> tuple[Path, dict]:
    dialect = req.dialect.upper()
    if dialect not in ALLOWED_DIALECTS:
        raise HTTPException(status_code=400, detail=f"Unsupported dialect: {req.dialect}")

    ckpt, vocab = _model_paths(req.model, dialect)
    output_name = req.output_file or f"habibi_{int(time.time() * 1000)}.wav"
    if "/" in output_name or not output_name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="output_file must be a simple .wav file name")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="infer-", dir=str(OUTPUT_DIR)))
    out_path = work_dir / output_name

    args = [
        "habibi-tts_infer-cli",
        "--model",
        req.model,
        "--dialect",
        dialect,
        "--ckpt_file",
        str(ckpt),
        "--vocab_file",
        str(vocab),
        "--ref_audio",
        req.ref_audio or DEFAULT_REF_AUDIO,
        "--ref_text",
        req.ref_text or DEFAULT_REF_TEXT,
        "--gen_text",
        req.text,
        "--output_dir",
        str(work_dir),
        "--output_file",
        out_path.name,
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
    ]
    if req.remove_silence:
        args.append("--remove_silence")
    if req.nfe_step is not None:
        args.extend(["--nfe_step", str(req.nfe_step)])
    if req.speed is not None:
        args.extend(["--speed", str(req.speed)])

    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd="/app",
        capture_output=True,
        text=True,
        timeout=req.timeout_seconds,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Habibi inference failed",
                "returncode": proc.returncode,
                "latency_ms": latency_ms,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            },
        )

    if not out_path.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Habibi inference completed but output file was not found",
                "latency_ms": latency_ms,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            },
        )

    meta = {
        "latency_ms": latency_ms,
        "bytes": out_path.stat().st_size,
        "model": req.model,
        "dialect": dialect,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "stdout": proc.stdout[-2000:],
    }
    return out_path, meta


@app.post("/infer")
def infer(req: InferRequest):
    out_path, meta = _run_inference(req)
    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename=out_path.name,
        headers={
            "x-habibi-latency-ms": str(meta["latency_ms"]),
            "x-habibi-model": str(meta["model"]),
            "x-habibi-dialect": str(meta["dialect"]),
            "x-habibi-device": str(meta["device"]),
        },
    )


@app.post("/infer_json")
def infer_json(req: InferRequest) -> dict:
    out_path, meta = _run_inference(req)
    audio_b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
    return {
        **meta,
        "format": "wav",
        "audio_base64": audio_b64,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
