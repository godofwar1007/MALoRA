"""
download_checkpoint.py — pull a specific checkpoint from HF Hub to resume training

Usage:
    python download_checkpoint.py --folder "100k_1ep_atoff/checkpoint-500"
    python download_checkpoint.py --folder "100k_1ep_atoff/checkpoint-500" --out ./outputs/malora/checkpoint-500
"""

import argparse
import os
from huggingface_hub import snapshot_download

HF_TOKEN   = "HF_TOKEN"   # ← update this
HF_REPO_ID = "godofwar1007/moelora"

def download(hf_folder: str, local_out: str):
    print(f"Downloading {HF_REPO_ID}/{hf_folder} → {local_out}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        allow_patterns=[f"{hf_folder}/*"],
        local_dir="./hf_download_tmp",
    )
    # move from nested path to target output dir
    import shutil
    nested = os.path.join("./hf_download_tmp", hf_folder)
    if os.path.exists(nested):
        os.makedirs(os.path.dirname(local_out), exist_ok=True)
        if os.path.exists(local_out):
            shutil.rmtree(local_out)
        shutil.move(nested, local_out)
        shutil.rmtree("./hf_download_tmp", ignore_errors=True)
    else:
        print(f"ERROR: expected path {nested} not found")
        print(f"Contents: {os.listdir('./hf_download_tmp')}")
        return

    print(f"Done. Checkpoint at: {local_out}")
    print(f"Files: {os.listdir(local_out)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="HF subfolder e.g. '100k_1ep_atoff/checkpoint-500'")
    parser.add_argument("--out", default=None, help="local output path (default: ./outputs/malora/checkpoint-NNN)")
    args = parser.parse_args()

    # auto-derive local output path from folder name if not specified
    checkpoint_name = args.folder.split("/")[-1]   # e.g. "checkpoint-500"
    local_out = args.out or f"./outputs/malora/{checkpoint_name}"

    download(args.folder, local_out)
