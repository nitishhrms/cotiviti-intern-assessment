"""Push model artifacts to HuggingFace Hub and update the model card."""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "xray_caption_models"


def push_to_hub(repo_id: str, hf_token: str, commit_message: str = "Update model"):
    try:
        from huggingface_hub import HfApi, upload_file
    except ImportError:
        print("ERROR: Install huggingface_hub:  pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=hf_token)

    # Ensure repository exists
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    files_to_push = [
        (MODEL_DIR / "best_model.pth", "best_model.pth"),
        (BASE_DIR / "vocab.json", "vocab.json"),
        (MODEL_DIR / "config.json", "config.json"),
        (BASE_DIR / "hf_model_card.md", "README.md"),
    ]

    # Also push ONNX if it exists
    onnx_path = MODEL_DIR / "xray_caption_model.onnx"
    if onnx_path.exists():
        files_to_push.append((onnx_path, "xray_caption_model.onnx"))

    for local_path, remote_path in files_to_push:
        if not local_path.exists():
            print(f"  SKIP {local_path} (not found)")
            continue
        print(f"  Uploading {local_path} → {repo_id}/{remote_path} …")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=commit_message,
        )

    print(f"\nDone. View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/xray-report-generator")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF API token")
    parser.add_argument("--message", default="Update model artifacts")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: HF_TOKEN env var or --token is required.")
        sys.exit(1)

    push_to_hub(args.repo_id, args.token, args.message)
