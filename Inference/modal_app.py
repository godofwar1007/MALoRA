import os
import modal
from pathlib import Path
from modal import FilePatternMatcher

HF_FOLDER  = "non-coder-instruct-2"
ATTN_ON    = True
MODEL_NAME = "malora"
APP_NAME   = "malora-server"
GPU_CONFIG = "" # subject to change 

hf_secret = modal.Secret.from_name("malora-hf-secret")

# Image Definition 

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.1.0",
        "transformers==4.47.0",   
        "accelerate>=0.26.0",
        "fastapi[standard]>=0.115.0",
        "uvicorn>=0.27.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0",
        "python-dotenv>=1.0.0",
        "sentencepiece>=0.1.99",   
        "packaging>=23.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "hf_transfer>=0.1.0",
    )
    .env({
        "PYTHONPATH":          "/root/code",
        "MALORA_HF_FOLDER":    HF_FOLDER,
        "MALORA_ATTN_ON":      str(int(ATTN_ON)),
        "MALORA_MODEL_NAME":   MODEL_NAME,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })

    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root/code",
        ignore=FilePatternMatcher.from_file(".modalignore"),
    )
)

# App 
app = modal.App(
    name=APP_NAME,
    image=image,
)

# Volume for model caching
model_volume = modal.Volume.from_name("malora-model-cache", create_if_missing=True)
CACHE_DIR    = "/root/code/hf_cache"

#ASGI Server Function

@app.function(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: model_volume},
    timeout=1800,
    secrets=[hf_secret],
.
    min_containers=0,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def serve():
    import os
    import sys
    os.chdir("/root/code")
    sys.path.insert(0, "/root/code")

    import server
    return server.app
