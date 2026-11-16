"""Download P-H3-Image quantized weights optimized for 24GB VRAM GPUs (e.g. RTX 3090)."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download


REPO_ID = "PrunaAI/P-H3-Image-Comfy"

PROFILES = {
    # Optimal for 24GB VRAM (RTX 3090 / 4090): ~20GB base + 9.7GB int8 upsampler (~30GB disk), peaks at ~21GB VRAM!
    "24gb": [
        "diffusion_models/p-h3-image_w4a8.safetensors",
        "text_encoders/p-h3-image_text_encoder_w4a8.safetensors",
        "text_encoders/p-h3-image_prompt_upsampler_int8_convrot.safetensors",
        "vae/p-h3-image_vae_int8_convrot.safetensors",
    ],
    # Smallest possible download (~27GB disk, peaks at ~21GB VRAM)
    "24gb-lite": [
        "diffusion_models/p-h3-image_w4a8.safetensors",
        "text_encoders/p-h3-image_text_encoder_w4a8.safetensors",
        "text_encoders/p-h3-image_prompt_upsampler_w4a8.safetensors",
        "vae/p-h3-image_vae_int8_convrot.safetensors",
    ],
    # Official 24GB mix with BF16 upsampler (~39GB disk, peaks at ~21GB VRAM)
    "24gb-bf16-upsampler": [
        "diffusion_models/p-h3-image_w4a8.safetensors",
        "text_encoders/p-h3-image_text_encoder_w4a8.safetensors",
        "text_encoders/p-h3-image_prompt_upsampler_bf16.safetensors",
        "vae/p-h3-image_vae_int8_convrot.safetensors",
    ],
    # For 32GB+ VRAM (RTX 5090 etc.): Full INT8 stack
    "int8": [
        "diffusion_models/p-h3-image_int8_convrot.safetensors",
        "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
        "text_encoders/p-h3-image_prompt_upsampler_int8_convrot.safetensors",
        "vae/p-h3-image_vae_int8_convrot.safetensors",
    ],
}


def check_missing_files(models_dir: Path, files: list[str]) -> list[str]:
    missing = []
    for rel_path in files:
        target = models_dir / rel_path
        if not target.exists() or target.stat().st_size < 1024 * 1024:
            missing.append(rel_path)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Download P-H3-Image models for ComfyUI backend.")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="24gb",
        help="Model precision profile to download (default: 24gb)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ComfyUI" / "models",
        help="Target ComfyUI models directory",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if required files exist; exit 0 if present, 1 if missing.",
    )
    args = parser.parse_args()

    files = PROFILES[args.profile]
    missing = check_missing_files(args.models_dir, files)

    if args.check_only:
        if missing:
            print(f"[INFO] Missing model files ({len(missing)}): {', '.join(missing)}")
            return 1
        print("[OK] All required P-H3-Image model files are present.")
        return 0

    if not missing:
        print(f"[OK] All models for profile '{args.profile}' are already downloaded in {args.models_dir}")
        return 0

    print(f"============================================================")
    print(f" Downloading P-H3-Image ({args.profile} profile)")
    print(f" Repository : {REPO_ID}")
    print(f" Target Dir : {args.models_dir}")
    print(f" Missing    : {len(missing)} file(s)")
    for m in missing:
        print(f"   - {m}")
    print(f"============================================================")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=missing,
        local_dir=str(args.models_dir),
    )

    still_missing = check_missing_files(args.models_dir, files)
    if still_missing:
        print(f"[ERROR] Failed to download: {still_missing}", file=sys.stderr)
        return 1

    print("[OK] Model download completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
