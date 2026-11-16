"""
P-H3-Image WebUI (A1111-style Gradio Frontend)
Automatically manages a headless ComfyUI backend and exposes a clean, intuitive
tab-based interface for Text-to-Image (txt2img), Image Editing (img2img + multi-ref),
and Gallery history.

Powered by MiniMax H3 / PrunaAI P-H3-Image.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import io
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import requests
import websocket
from PIL import Image, PngImagePlugin

ROOT_DIR = Path(__file__).resolve().parent
COMFY_DIR = ROOT_DIR / "ComfyUI"
MODELS_DIR = COMFY_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_T2I_DIR = OUTPUTS_DIR / "txt2img"
OUTPUTS_EDIT_DIR = OUTPUTS_DIR / "img2img_edit"

for d in (OUTPUTS_DIR, OUTPUTS_T2I_DIR, OUTPUTS_EDIT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"

ASPECT_PRESETS_8MP: dict[str, tuple[int, int, str]] = {
    "1:1 (2912 × 2912) [推奨 8MP]": (2912, 2912, "1:1"),
    "4:3 (3360 × 2496) [8MP 横]": (3360, 2496, "4:3"),
    "3:4 (2496 × 3360) [8MP 縦]": (2496, 3360, "3:4"),
    "3:2 (3552 × 2368) [8MP 写真横]": (3552, 2368, "3:2"),
    "2:3 (2368 × 3552) [8MP 写真縦]": (2368, 3552, "2:3"),
    "16:9 (3872 × 2176) [8MP ワイド横]": (3872, 2176, "16:9"),
    "9:16 (2176 × 3872) [8MP スマホ縦]": (2176, 3872, "9:16"),
    "1:1 (2048 × 2048) [軽量 4MP]": (2048, 2048, "1:1"),
    "4:3 (2368 × 1760) [軽量 4MP 横]": (2368, 1760, "4:3"),
    "3:4 (1760 × 2368) [軽量 4MP 縦]": (1760, 2368, "3:4"),
    "16:9 (2720 × 1536) [軽量 4MP ワイド]": (2720, 1536, "16:9"),
    "9:16 (1536 × 2720) [軽量 4MP 縦ワイド]": (1536, 2720, "9:16"),
    "カスタム (手動指定)": (0, 0, "custom"),
}


def closest_aspect_label(width: int, height: int) -> str:
    """Return the closest canonical aspect ratio string expected by P-H3-Image upsampler."""
    candidates = [
        ("1:1", 1.0),
        ("4:3", 4.0 / 3.0),
        ("3:4", 3.0 / 4.0),
        ("3:2", 3.0 / 2.0),
        ("2:3", 2.0 / 3.0),
        ("16:9", 16.0 / 9.0),
        ("9:16", 9.0 / 16.0),
    ]
    ratio = max(width, 1) / max(height, 1)
    best_name, _ = min(candidates, key=lambda item: abs(math.log(ratio / item[1])))
    return best_name


def list_model_files(subdir: str, prefix_filter: str = "") -> list[str]:
    folder = MODELS_DIR / subdir
    if not folder.exists():
        return []
    files = sorted(
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix in (".safetensors", ".ckpt", ".pt", ".gguf")
        and (not prefix_filter or prefix_filter in f.name)
    )
    return files


def pick_default(options: list[str], preferred_substrings: list[str], fallback: str) -> str:
    for sub in preferred_substrings:
        for opt in options:
            if sub in opt:
                return opt
    return options[0] if options else fallback


class ComfyBackendManager:
    """Manages the local headless ComfyUI server process and API communication."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8188, auto_launch: bool = True):
        self.host = host
        self.port = port
        self.auto_launch = auto_launch
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.client_id = str(uuid.uuid4())
        atexit.register(self.stop)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"

    def is_ready(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def ensure_running(self, progress: gr.Progress | None = None) -> None:
        if self.is_ready():
            return

        with self._lock:
            if self.is_ready():
                return
            if not self.auto_launch:
                raise RuntimeError(
                    f"ComfyUI バックエンド ({self.base_url}) に接続できません。"
                )

            main_py = COMFY_DIR / "main.py"
            if not main_py.exists():
                raise RuntimeError(f"ComfyUI が見つかりません: {main_py}")

            # Sync bundled helper custom nodes into ComfyUI/custom_nodes/
            bundled_nodes_dir = ROOT_DIR / "custom_nodes"
            target_nodes_dir = COMFY_DIR / "custom_nodes"
            if bundled_nodes_dir.exists() and target_nodes_dir.exists():
                for py_file in bundled_nodes_dir.glob("*.py"):
                    shutil.copy2(py_file, target_nodes_dir / py_file.name)

            if progress is not None:
                progress(0.02, desc="推論エンジン (ComfyUI Backend) をバックグラウンド起動中...")

            print(f"[Backend] Starting headless ComfyUI server on {self.host}:{self.port} ...")
            cmd = [
                sys.executable,
                str(main_py),
                "--listen",
                self.host,
                "--port",
                str(self.port),
                "--disable-auto-launch",
            ]
            # Launch ComfyUI cleanly in background, writing logs to comfyui_backend.log
            log_file_path = ROOT_DIR / "comfyui_backend.log"
            self._log_fh = open(log_file_path, "w", encoding="utf-8", buffering=1)
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and os.environ.get("PH3_SHOW_BACKEND_CONSOLE") != "1" else 0
            self.process = subprocess.Popen(
                cmd,
                cwd=str(COMFY_DIR),
                stdin=subprocess.DEVNULL,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

            deadline = time.time() + 180.0
            while time.time() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"推論エンジンの起動に失敗しました (exit code {self.process.returncode})。"
                    )
                if self.is_ready():
                    print(f"[Backend] ComfyUI engine is ready at {self.base_url}")
                    return
                time.sleep(1.0)

            raise RuntimeError("推論エンジンの起動がタイムアウトしました (180秒)。")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            print("[Backend] Stopping ComfyUI engine...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def interrupt(self) -> str:
        if not self.is_ready():
            return "エンジン未起動"
        try:
            requests.post(f"{self.base_url}/interrupt", timeout=5.0)
            return "⏹ 生成の中断リクエストを送信しました。"
        except Exception as e:
            return f"中断エラー: {e}"

    def free_vram(self) -> str:
        if not self.is_ready():
            return "エンジン未起動"
        try:
            requests.post(
                f"{self.base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=10.0,
            )
            return "🧹 VRAM・モデルキャッシュを解放しました。"
        except Exception as e:
            return f"VRAM解放エラー: {e}"

    def get_vram_status(self) -> str:
        if not self.is_ready():
            return "🟡 推論エンジン待機中（初回生成または起動時に自動接続）"
        try:
            data = requests.get(f"{self.base_url}/system_stats", timeout=3.0).json()
            devices = data.get("devices", [])
            if not devices:
                return "🟢 エンジン稼働中"
            dev = devices[0]
            name = dev.get("name", "GPU").replace("cuda:0 ", "")
            vram_total = dev.get("vram_total", 0) / (1024**3)
            vram_free = dev.get("vram_free", 0) / (1024**3)
            vram_used = max(0.0, vram_total - vram_free)
            return f"🟢 エンジン稼働中 | {name} | VRAM使用量: {vram_used:.1f} / {vram_total:.1f} GB"
        except Exception:
            return "🟢 エンジン稼働中"

    def upload_image(self, pil_img: Image.Image, prefix: str = "ph3_input") -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        filename = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        resp = requests.post(
            f"{self.base_url}/upload/image",
            files={"image": (filename, buf, "image/png")},
            data={"overwrite": "true", "type": "input"},
            timeout=30.0,
        )
        resp.raise_for_status()
        res_json = resp.json()
        subfolder = res_json.get("subfolder", "")
        name = res_json["name"]
        return f"{subfolder}/{name}" if subfolder else name

    def execute_workflow(
        self,
        prompt_graph: dict[str, Any],
        progress: gr.Progress | None = None,
        node_labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.ensure_running(progress=progress)
        node_labels = node_labels or {}

        ws = websocket.WebSocket()
        ws.settimeout(5.0)
        ws.connect(self.ws_url)

        try:
            payload = {"prompt": prompt_graph, "client_id": self.client_id}
            resp = requests.post(f"{self.base_url}/prompt", json=payload, timeout=30.0)
            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI API Error ({resp.status_code}): {resp.text}")
            prompt_id = resp.json()["prompt_id"]

            if progress is not None:
                progress(0.05, desc="ジョブを送信しました。モデルを準備中...")

            while True:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    # Periodically check history in case WS missed completion
                    hist_resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=5.0)
                    if hist_resp.status_code == 200 and prompt_id in hist_resp.json():
                        break
                    continue

                if isinstance(raw, bytes):
                    # Binary preview frame, ignore
                    continue

                msg = json.loads(raw)
                mtype = msg.get("type")
                mdata = msg.get("data", {})

                if mtype == "executing":
                    if mdata.get("prompt_id") == prompt_id:
                        node_id = mdata.get("node")
                        if node_id is None:
                            # Execution finished!
                            break
                        label = node_labels.get(str(node_id), f"ノード #{node_id} 実行中")
                        if progress is not None:
                            progress(0.15, desc=label)

                elif mtype == "progress":
                    if mdata.get("prompt_id") in (None, prompt_id):
                        val = mdata.get("value", 0)
                        max_val = max(mdata.get("max", 1), 1)
                        node_id = str(mdata.get("node", ""))
                        label = node_labels.get(node_id, "サンプリング中")
                        frac = 0.2 + 0.75 * (val / max_val)
                        if progress is not None:
                            progress(min(frac, 0.96), desc=f"{label} ({val}/{max_val})")

                elif mtype == "execution_error":
                    if mdata.get("prompt_id") == prompt_id:
                        err_msg = mdata.get("exception_message", "Unknown execution error")
                        node_type = mdata.get("node_type", "")
                        raise RuntimeError(f"生成エラー ({node_type}): {err_msg}")
        finally:
            try:
                ws.close()
            except Exception:
                pass

        # Fetch outputs from history
        hist = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=15.0).json()
        if prompt_id not in hist:
            raise RuntimeError("生成履歴の取得に失敗しました（中断された可能性があります）。")

        entry = hist[prompt_id]
        status_info = entry.get("status", {})
        if status_info.get("status_str") == "error":
            raise RuntimeError(f"推論エンジンでエラーが発生しました: {status_info.get('messages')}")

        outputs = entry.get("outputs", {})
        return outputs

    def fetch_image(self, img_meta: dict[str, str]) -> Image.Image:
        params = {
            "filename": img_meta["filename"],
            "subfolder": img_meta.get("subfolder", ""),
            "type": img_meta.get("type", "output"),
        }
        resp = requests.get(f"{self.base_url}/view", params=params, timeout=30.0)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ---------------------------------------------------------------------------
# API Graph Builders for P-H3-Image
# ---------------------------------------------------------------------------

def build_t2i_api_graph(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    use_upsampler: bool,
    unet_name: str,
    text_encoder_name: str,
    upsampler_name: str,
    vae_name: str,
    sampler_name: str = "euler",
    sigmas: str = DEFAULT_SIGMAS,
    shift_video: float = 1.0,
    shift_audio: float = 1.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    aspect_str = closest_aspect_label(width, height)

    graph: dict[str, Any] = {
        "22": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        "23": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["22", 0],
                "shift_video": shift_video,
                "shift_audio": shift_audio,
            },
        },
        "24": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder_name,
                "type": "minimax",
                "device": "default",
            },
        },
        "25": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        "15": {
            "class_type": "PreviewAny",
            "inputs": {
                "source": prompt,
            },
        },
        "26": {
            "class_type": "PH3ImageConditioning",
            "inputs": {
                "clip": ["24", 0],
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
            },
        },
        "6": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": sigmas,
            },
        },
        "7": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["23", 0],
                "conditioning": ["26", 0],
            },
        },
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": sampler_name,
            },
        },
        "9": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": int(seed),
            },
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["7", 0],
                "sampler": ["8", 0],
                "sigmas": ["6", 0],
                "latent_image": ["26", 1],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["25", 0],
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["11", 0],
                "filename_prefix": "p-h3-t2i",
            },
        },
    }

    if use_upsampler:
        upsampler_prompt = f"Aspect ratio: {aspect_str}\nRequest: {prompt}"
        graph["13"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": upsampler_name,
                "type": "minimax",
                "device": "default",
            },
        }
        graph["18"] = {
            "class_type": "TextGenerate",
            "inputs": {
                "clip": ["13", 0],
                "prompt": upsampler_prompt,
                "max_length": 2048,
                "sampling_mode": "off",
                "thinking": True,
                "use_default_template": True,
                "mtp": "auto",
            },
        }
        graph["15"]["inputs"]["source"] = ["18", 0]
        graph["26"]["inputs"]["prompt"] = ["18", 0]

    labels = {
        "13": "🧠 プロンプト拡張モデル (Qwen3.5) をロード中...",
        "18": "✍️ AIが短い指示から詳細プロンプトを生成中...",
        "22": "🎨 P-H3-Image 本体モデル (w4a8) をロード中...",
        "24": "📝 テキストエンコーダーをロード中...",
        "26": "📐 条件付け・Latentを準備中...",
        "10": "⚡ 画像を高速サンプリング中 (8ステップ)",
        "11": "🖼️ 高解像度画像をVAEデコード中...",
        "12": "💾 画像を保存中...",
    }
    return graph, labels


def build_edit_api_graph(
    prompt: str,
    uploaded_image_names: list[str],
    target_mp: float,
    custom_width: int,
    custom_height: int,
    seed: int,
    use_upsampler: bool,
    unet_name: str,
    text_encoder_name: str,
    upsampler_name: str,
    vae_name: str,
    lock_style: bool = True,
    lock_character: bool = True,
    denoise: float = 1.0,
    ref_cfg_boost: float = 1.0,
    sampler_name: str = "euler",
    sigmas: str = DEFAULT_SIGMAS,
    shift_video: float = 1.0,
    shift_audio: float = 1.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not uploaded_image_names:
        raise ValueError("編集元の画像（Picture 1）を最低1枚アップロードしてください。")

    graph: dict[str, Any] = {
        "26": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        "27": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["26", 0],
                "shift_video": shift_video,
                "shift_audio": shift_audio,
            },
        },
        "28": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder_name,
                "type": "minimax",
                "device": "default",
            },
        },
        "29": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        "7": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": sigmas,
            },
        },
        "8": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["27", 0],
                "conditioning": ["30", 0],
            },
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": sampler_name,
            },
        },
        "10": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": int(seed),
            },
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["8", 0],
                "sampler": ["9", 0],
                "sigmas": ["7", 0],
                "latent_image": ["30", 1],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["29", 0],
            },
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["12", 0],
                "filename_prefix": "p-h3-edit",
            },
        },
    }

    # Optional Reference Anchor Guidance Boost (DualModelGuider with image-only null negative)
    if float(ref_cfg_boost) > 1.001:
        graph["8"] = {
            "class_type": "DualModelGuider",
            "inputs": {
                "model": ["27", 0],
                "positive": ["30", 0],
                "cfg": float(ref_cfg_boost),
            },
        }

    # Load all input images (1..8)
    for idx, img_name in enumerate(uploaded_image_names, start=1):
        graph[f"load_{idx}"] = {
            "class_type": "LoadImage",
            "inputs": {"image": img_name},
        }

    # Scale image_1 to target megapixels (default 4 MP) for conditioning
    graph["31"] = {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {
            "image": ["load_1", 0],
            "upscale_method": "lanczos",
            "megapixels": float(target_mp),
            "resolution_steps": 32,
        },
    }

    # Optional classic A1111-style img2img latent blending when denoise < 1.0
    if float(denoise) < 0.999:
        graph["32"] = {
            "class_type": "PH3ImageToAVLatent",
            "inputs": {
                "vae": ["29", 0],
                "image": ["31", 0],
                "denoise": float(denoise),
                "sigmas": ["7", 0],
            },
        }
        graph["11"]["inputs"]["latent_image"] = ["32", 0]
        graph["11"]["inputs"]["sigmas"] = ["32", 1]

    # Build strict constraint suffix when lock toggles are active
    lock_directives: list[str] = []
    if lock_style:
        lock_directives.append(
            "CRITICAL ART STYLE LOCK: Preserve the exact artistic medium, linework texture, coloring/brushwork technique (e.g., watercolor, colored pencil, oil, sketch, cel, or photo), paper/canvas texture, and shading softness of <Picture 1>. NEVER substitute a generic glossy digital anime style."
        )
    if lock_character:
        lock_directives.append(
            "CRITICAL CHARACTER LOCK: Preserve <Picture 1>'s exact character identity: exact hair silhouette, how the hair is tied (e.g., low twin buns vs long ponytails), hair ribbons/accessories, bangs, exact eye shape/iris style, facial proportions, age impression, and body build."
        )

    augmented_req = prompt.strip()
    if lock_directives:
        augmented_req = f"{augmented_req}\n\n" + "\n".join(lock_directives)

    cond_inputs: dict[str, Any] = {
        "clip": ["28", 0],
        "vae": ["29", 0],
        "prompt": augmented_req,
        "width": int(custom_width),
        "height": int(custom_height),
        "images.image_1": ["31", 0],
    }
    for idx in range(2, len(uploaded_image_names) + 1):
        cond_inputs[f"images.image_{idx}"] = [f"load_{idx}", 0]

    graph["16"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": augmented_req},
    }

    if use_upsampler:
        # Scale image_1 to 1 MP for vision upsampler
        graph["19"] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["load_1", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 1,
            },
        }
        batch_inputs: dict[str, Any] = {
            "images.image0": ["19", 0],
        }
        if len(uploaded_image_names) > 1:
            graph["20"] = {
                "class_type": "GetImageSize",
                "inputs": {"image": ["19", 0]},
            }
            for idx in range(2, len(uploaded_image_names) + 1):
                pad_id = f"pad_{idx}"
                graph[pad_id] = {
                    "class_type": "ResizeAndPadImage",
                    "inputs": {
                        "image": [f"load_{idx}", 0],
                        "target_width": ["20", 0],
                        "target_height": ["20", 1],
                        "padding_color": "white",
                        "interpolation": "lanczos",
                    },
                }
                batch_inputs[f"images.image{idx - 1}"] = [pad_id, 0]

        graph["22"] = {
            "class_type": "BatchImagesNode",
            "inputs": batch_inputs,
        }
        graph["14"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": upsampler_name,
                "type": "minimax",
                "device": "default",
            },
        }

        pic_tags = "".join(
            f"<Picture {i}>: <|vision_start|><|image_pad|><|vision_end|>"
            for i in range(1, len(uploaded_image_names) + 1)
        )
        upsampler_input_text = (
            f"<|im_start|>user\n{pic_tags}Edit <Picture 1>. {augmented_req}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        graph["23"] = {
            "class_type": "TextGenerate",
            "inputs": {
                "clip": ["14", 0],
                "image": ["22", 0],
                "prompt": upsampler_input_text,
                "max_length": 3072,
                "sampling_mode": "off",
                "thinking": False,
                "use_default_template": True,
                "mtp": "auto",
            },
        }
        graph["24_strict"] = {
            "class_type": "PH3StrictBriefPostprocess",
            "inputs": {
                "brief": ["23", 0],
                "lock_style": bool(lock_style),
                "lock_character": bool(lock_character),
            },
        }
        cond_inputs["prompt"] = ["24_strict", 0]
        graph["16"]["inputs"]["source"] = ["24_strict", 0]

    graph["30"] = {
        "class_type": "PH3ImageConditioning",
        "inputs": cond_inputs,
    }

    labels = {
        "14": "🧠 画像認識プロンプト拡張モデル (Qwen3.5) をロード中...",
        "23": "👁️ 入力画像の画風・キャラ特徴を精密解析中...",
        "26": "🎨 P-H3-Image 本体モデル (w4a8) をロード中...",
        "28": "📝 テキストエンコーダーをロード中...",
        "30": "📐 参照画像のVAEエンコード・条件付け中...",
        "32": "🖼️ 元画像Latentをブレンド中 (img2img Denoise)...",
        "11": "⚡ 画像編集サンプリング中",
        "12": "🖼️ 編集結果をVAEデコード中...",
        "13": "💾 画像を保存中...",
    }
    return graph, labels


# ---------------------------------------------------------------------------
# Saving & Metadata Helpers
# ---------------------------------------------------------------------------

def save_output_with_infotext(
    img: Image.Image,
    save_dir: Path,
    prefix: str,
    infotext: str,
) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    out_path = save_dir / f"{prefix}_{timestamp}_{short_id}.png"

    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", infotext)
    meta.add_text("Generator", "P-H3-Image WebUI (Powered by MiniMax H3)")
    img.save(out_path, format="PNG", pnginfo=meta)
    return out_path


def list_gallery_images(limit: int = 60) -> list[str]:
    files = sorted(
        OUTPUTS_DIR.rglob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [str(p) for p in files[:limit]]


# ---------------------------------------------------------------------------
# Gradio WebUI Application
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    max-width: 1560px !important;
}
.generate-btn {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.12rem !important;
    min-height: 62px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.35) !important;
}
.generate-btn:hover {
    filter: brightness(1.07);
}
.stop-btn {
    min-height: 62px !important;
    font-weight: 600 !important;
}
.status-bar {
    font-size: 0.92rem;
    padding: 6px 12px;
    border-radius: 8px;
}
.footer-badge {
    text-align: center;
    opacity: 0.78;
    font-size: 0.85rem;
    margin-top: 16px;
}
"""


def create_ui(backend: ComfyBackendManager) -> gr.Blocks:
    unet_choices = list_model_files("diffusion_models") or ["p-h3-image_w4a8.safetensors"]
    te_choices = list_model_files("text_encoders", "text_encoder") or ["p-h3-image_text_encoder_w4a8.safetensors"]
    up_choices = list_model_files("text_encoders", "prompt_upsampler") or ["p-h3-image_prompt_upsampler_int8_convrot.safetensors"]
    vae_choices = list_model_files("vae") or ["p-h3-image_vae_int8_convrot.safetensors"]

    default_unet = pick_default(unet_choices, ["w4a8", "int8", "bf16"], "p-h3-image_w4a8.safetensors")
    default_te = pick_default(te_choices, ["w4a8", "int8", "bf16"], "p-h3-image_text_encoder_w4a8.safetensors")
    default_up = pick_default(up_choices, ["int8", "bf16", "w4a8"], "p-h3-image_prompt_upsampler_int8_convrot.safetensors")
    default_vae = pick_default(vae_choices, ["int8", "bf16"], "p-h3-image_vae_int8_convrot.safetensors")

    def refresh_models():
        u = list_model_files("diffusion_models") or ["p-h3-image_w4a8.safetensors"]
        t = list_model_files("text_encoders", "text_encoder") or ["p-h3-image_text_encoder_w4a8.safetensors"]
        p = list_model_files("text_encoders", "prompt_upsampler") or ["p-h3-image_prompt_upsampler_int8_convrot.safetensors"]
        v = list_model_files("vae") or ["p-h3-image_vae_int8_convrot.safetensors"]
        return (
            gr.update(choices=u, value=pick_default(u, ["w4a8", "int8", "bf16"], u[0])),
            gr.update(choices=t, value=pick_default(t, ["w4a8", "int8", "bf16"], t[0])),
            gr.update(choices=p, value=pick_default(p, ["int8", "bf16", "w4a8"], p[0])),
            gr.update(choices=v, value=pick_default(v, ["int8", "bf16"], v[0])),
            backend.get_vram_status(),
        )

    def on_preset_change(preset_key: str):
        w, h, _ = ASPECT_PRESETS_8MP.get(preset_key, (2912, 2912, "1:1"))
        if w == 0 and h == 0:
            return gr.update(interactive=True), gr.update(interactive=True)
        return gr.update(value=w, interactive=True), gr.update(value=h, interactive=True)

    def run_t2i(
        prompt: str,
        width: int,
        height: int,
        seed: int,
        batch_count: int,
        use_upsampler: bool,
        sampler_name: str,
        sigmas: str,
        shift_video: float,
        shift_audio: float,
        unet_name: str,
        te_name: str,
        up_name: str,
        vae_name: str,
        progress=gr.Progress(),
    ):
        if not prompt or not prompt.strip():
            raise gr.Error("プロンプトを入力してください。")

        width = max(256, (int(width) // 32) * 32)
        height = max(256, (int(height) // 32) * 32)
        batch_count = max(1, int(batch_count))
        base_seed = random.randint(1, 2**53 - 1) if int(seed) == -1 else int(seed)

        generated_files: list[str] = []
        last_upsampled = ""
        last_seed = base_seed

        for b in range(batch_count):
            cur_seed = base_seed + b
            last_seed = cur_seed
            if batch_count > 1:
                progress(0.01, desc=f"バッチ {b + 1}/{batch_count} を開始中...")

            graph, labels = build_t2i_api_graph(
                prompt=prompt.strip(),
                width=width,
                height=height,
                seed=cur_seed,
                use_upsampler=use_upsampler,
                unet_name=unet_name,
                text_encoder_name=te_name,
                upsampler_name=up_name,
                vae_name=vae_name,
                sampler_name=sampler_name,
                sigmas=sigmas.strip(),
                shift_video=float(shift_video),
                shift_audio=float(shift_audio),
            )

            outputs = backend.execute_workflow(graph, progress=progress, node_labels=labels)

            # Extract upsampled prompt from PreviewAny node "15"
            if "15" in outputs and "text" in outputs["15"]:
                texts = outputs["15"]["text"]
                if texts:
                    last_upsampled = texts[0]

            # Extract saved image(s) from SaveImage node "12"
            if "12" not in outputs or "images" not in outputs["12"]:
                raise gr.Error("画像出力が取得できませんでした。")

            for img_meta in outputs["12"]["images"]:
                pil_img = backend.fetch_image(img_meta)
                infotext = (
                    f"{prompt.strip()}\n"
                    f"Upsampled prompt: {last_upsampled}\n"
                    f"Steps: 8, Sampler: {sampler_name}, Size: {pil_img.width}x{pil_img.height}, "
                    f"Seed: {cur_seed}, Model: {unet_name}, Upsampler: {up_name if use_upsampler else 'Off'}"
                )
                saved_path = save_output_with_infotext(pil_img, OUTPUTS_T2I_DIR, "t2i", infotext)
                generated_files.append(str(saved_path))

        progress(1.0, desc="完了！")
        info_md = (
            f"**Seed:** `{last_seed}` &nbsp;|&nbsp; **解像度:** `{width} × {height}` "
            f"({(width * height) / 1e6:.2f} MP) &nbsp;|&nbsp; **保存先:** `outputs/txt2img/`"
        )
        return generated_files, last_upsampled, last_seed, info_md, backend.get_vram_status()

    def run_edit(
        prompt: str,
        base_image: Image.Image | None,
        ref_gallery: list[Any] | None,
        lock_style: bool,
        lock_character: bool,
        denoise: float,
        ref_cfg_boost: float,
        target_mp: float,
        custom_w: int,
        custom_h: int,
        seed: int,
        batch_count: int,
        use_upsampler: bool,
        sampler_name: str,
        sigmas: str,
        shift_video: float,
        shift_audio: float,
        unet_name: str,
        te_name: str,
        up_name: str,
        vae_name: str,
        progress=gr.Progress(),
    ):
        if base_image is None:
            raise gr.Error("編集対象の画像（ベース画像 / Picture 1）をアップロードしてください。")
        if not prompt or not prompt.strip():
            raise gr.Error("編集指示（プロンプト）を入力してください。")

        backend.ensure_running(progress=progress)
        progress(0.04, desc="入力画像をエンジンへ転送中...")

        pil_images: list[Image.Image] = [base_image.convert("RGB")]
        if ref_gallery:
            for item in ref_gallery[:7]:
                # Gradio Gallery item can be filepath str or (filepath, caption) tuple or PIL Image
                path_or_img = item[0] if isinstance(item, (tuple, list)) else item
                if isinstance(path_or_img, Image.Image):
                    pil_images.append(path_or_img.convert("RGB"))
                else:
                    pil_images.append(Image.open(path_or_img).convert("RGB"))

        uploaded_names = [
            backend.upload_image(img, prefix=f"ph3_pic{i}")
            for i, img in enumerate(pil_images, start=1)
        ]

        cw = (int(custom_w) // 32) * 32 if int(custom_w) > 0 else 0
        ch = (int(custom_h) // 32) * 32 if int(custom_h) > 0 else 0
        batch_count = max(1, int(batch_count))
        base_seed = random.randint(1, 2**53 - 1) if int(seed) == -1 else int(seed)

        generated_files: list[str] = []
        last_upsampled = ""
        last_seed = base_seed

        for b in range(batch_count):
            cur_seed = base_seed + b
            last_seed = cur_seed

            graph, labels = build_edit_api_graph(
                prompt=prompt.strip(),
                uploaded_image_names=uploaded_names,
                target_mp=float(target_mp),
                custom_width=cw,
                custom_height=ch,
                seed=cur_seed,
                use_upsampler=use_upsampler,
                unet_name=unet_name,
                text_encoder_name=te_name,
                upsampler_name=up_name,
                vae_name=vae_name,
                lock_style=bool(lock_style),
                lock_character=bool(lock_character),
                denoise=float(denoise),
                ref_cfg_boost=float(ref_cfg_boost),
                sampler_name=sampler_name,
                sigmas=sigmas.strip(),
                shift_video=float(shift_video),
                shift_audio=float(shift_audio),
            )

            outputs = backend.execute_workflow(graph, progress=progress, node_labels=labels)

            if "16" in outputs and "text" in outputs["16"]:
                texts = outputs["16"]["text"]
                if texts:
                    last_upsampled = texts[0]

            if "13" not in outputs or "images" not in outputs["13"]:
                raise gr.Error("編集画像の出力が取得できませんでした。")

            for img_meta in outputs["13"]["images"]:
                pil_img = backend.fetch_image(img_meta)
                infotext = (
                    f"Edit Instruction: {prompt.strip()}\n"
                    f"LockStyle: {lock_style}, LockChar: {lock_character}, Denoise: {denoise}, RefBoost: {ref_cfg_boost}\n"
                    f"Upsampled brief: {last_upsampled}\n"
                    f"Input Pictures: {len(uploaded_names)}, Steps: 8, Sampler: {sampler_name}, "
                    f"Size: {pil_img.width}x{pil_img.height}, Seed: {cur_seed}, Model: {unet_name}"
                )
                saved_path = save_output_with_infotext(pil_img, OUTPUTS_EDIT_DIR, "edit", infotext)
                generated_files.append(str(saved_path))

        progress(1.0, desc="完了！")
        mode_tag = "Image-Edit (構図フリー)" if float(denoise) >= 0.999 else f"img2img (Denoise {denoise:.2f})"
        info_md = (
            f"**Seed:** `{last_seed}` &nbsp;|&nbsp; **モード:** `{mode_tag}` &nbsp;|&nbsp; "
            f"**ガチ固定:** `画風={'ON' if lock_style else 'OFF'} / キャラ={'ON' if lock_character else 'OFF'}` "
            f"&nbsp;|&nbsp; **参照:** `{len(uploaded_names)}枚`"
        )
        return generated_files, last_upsampled, last_seed, info_md, backend.get_vram_status()

    def send_gallery_to_edit(gallery_items: list[Any] | None):
        if not gallery_items:
            raise gr.Error("送る画像がありません。先に画像を生成してください。")
        first = gallery_items[0]
        path = first[0] if isinstance(first, (tuple, list)) else first
        img = Image.open(path).convert("RGB") if not isinstance(path, Image.Image) else path
        return img, gr.Tabs(selected="tab_edit")

    def open_outputs_folder():
        try:
            if sys.platform == "win32":
                os.startfile(str(OUTPUTS_DIR))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(OUTPUTS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(OUTPUTS_DIR)])
            return f"📂 フォルダを開きました: {OUTPUTS_DIR}"
        except Exception as e:
            return f"フォルダパス: {OUTPUTS_DIR} ({e})"

    with gr.Blocks(title="PrismH3 WebUI — P-H3-Image Studio") as demo:
        # Top Header & Checkpoint Bar (A1111 style)
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown(
                    "## 🎨 PrismH3 WebUI — P-H3-Image Studio\n"
                    "ノード配線不要・A1111ライクに使える **P-H3-Image** 専用インターフェース（8ステップ高速生成 / 最大8MP / ガチ固定マルチ参照編集）"
                )
            with gr.Column(scale=4):
                status_box = gr.Markdown(value=backend.get_vram_status(), elem_classes=["status-bar"])
                with gr.Row():
                    refresh_btn = gr.Button("🔄 ステータス・モデル更新", size="sm")
                    free_vram_btn = gr.Button("🧹 VRAM解放", size="sm")
                    open_dir_btn = gr.Button("📂 保存フォルダを開く", size="sm")

        with gr.Accordion("⚙️ 使用モデル設定（RTX 3090 24GB 最適化セット自動選択済み）", open=False):
            with gr.Row():
                unet_dd = gr.Dropdown(
                    choices=unet_choices,
                    value=default_unet,
                    label="Diffusion Model (本体)",
                )
                te_dd = gr.Dropdown(
                    choices=te_choices,
                    value=default_te,
                    label="Text Encoder",
                )
                up_dd = gr.Dropdown(
                    choices=up_choices,
                    value=default_up,
                    label="Prompt Upsampler (Qwen3.5-9B)",
                )
                vae_dd = gr.Dropdown(
                    choices=vae_choices,
                    value=default_vae,
                    label="VAE",
                )

        # Main Tabs
        with gr.Tabs() as main_tabs:
            # ===============================================================
            # TAB 1: txt2img
            # ===============================================================
            with gr.TabItem("🖼️ txt2img (画像生成)", id="tab_t2i"):
                with gr.Row():
                    with gr.Column(scale=8):
                        t2i_prompt = gr.Textbox(
                            label="プロンプト (Prompt) — 日本語でも短い英語でもOK！AIアップサンプラーが自動で詳細化します",
                            placeholder='例: 雨の夜の東京の路地裏、「OPEN 24/7」と書かれたネオンサインが輝くラーメン屋、シネマティックな実写写真',
                            lines=4,
                        )
                    with gr.Column(scale=3):
                        with gr.Row():
                            t2i_gen_btn = gr.Button("🚀 生成 (Generate)", variant="primary", elem_classes=["generate-btn"], scale=3)
                            t2i_stop_btn = gr.Button("⏹ 中断", variant="stop", elem_classes=["stop-btn"], scale=1)

                with gr.Row():
                    # Left Column: Parameters (A1111 layout)
                    with gr.Column(scale=5):
                        t2i_preset = gr.Dropdown(
                            choices=list(ASPECT_PRESETS_8MP.keys()),
                            value="1:1 (2912 × 2912) [推奨 8MP]",
                            label="📐 解像度プリセット (P-H3-Image は 8MP が最も高画質)",
                        )
                        with gr.Row():
                            t2i_width = gr.Slider(
                                minimum=512, maximum=4096, step=32, value=2912, label="幅 (Width)"
                            )
                            t2i_height = gr.Slider(
                                minimum=512, maximum=4096, step=32, value=2912, label="高さ (Height)"
                            )
                            t2i_swap_btn = gr.Button("⇅ 縦横入替", size="sm", scale=0)

                        t2i_use_upsampler = gr.Checkbox(
                            value=True,
                            label="✨ プロンプト自動拡張 (Prompt Upsampler / Qwen3.5) を使う【強く推奨】",
                            info="ONにすると、1行の短い指示や日本語でもP-H3-Imageに最適化された英語長文プロンプトへ自動変換してから描画します。",
                        )

                        with gr.Row():
                            t2i_seed = gr.Number(
                                value=-1, precision=0, label="🎲 シード値 (Seed: -1 でランダム)"
                            )
                            t2i_random_btn = gr.Button("🎲 -1に戻す", size="sm", scale=0)
                            t2i_reuse_btn = gr.Button("♻️ 直前のSeed", size="sm", scale=0)

                        t2i_batch = gr.Slider(
                            minimum=1, maximum=8, step=1, value=1, label="連続生成枚数 (Batch count)"
                        )

                        with gr.Accordion("🔧 詳細サンプリング設定 (通常は変更不要)", open=False):
                            gr.Markdown(
                                "*P-H3-Image は CFGなし（ネガティブプロンプト不要）・専用8ステップSigmaスケジュールで動作します。*"
                            )
                            t2i_sampler = gr.Dropdown(
                                choices=["euler", "euler_ancestral", "heun", "dpmpp_2m"],
                                value="euler",
                                label="Sampler",
                            )
                            t2i_sigmas = gr.Textbox(
                                value=DEFAULT_SIGMAS,
                                label="Manual Sigmas (8 steps)",
                            )
                            with gr.Row():
                                t2i_shift_v = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Shift Video")
                                t2i_shift_a = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Shift Audio")

                    # Right Column: Output Preview
                    with gr.Column(scale=6):
                        t2i_gallery = gr.Gallery(
                            label="生成結果",
                            columns=1,
                            height=620,
                            object_fit="contain",
                            preview=True,
                        )
                        t2i_info = gr.Markdown("準備完了")
                        with gr.Row():
                            send_to_edit_btn = gr.Button("🖌️ この画像を「画像編集 (img2img)」タブへ送る", size="sm")
                        with gr.Accordion("📝 AIが拡張した実際のプロンプトを確認", open=False):
                            t2i_upsampled_out = gr.Textbox(
                                label="Upsampled Prompt", lines=5, interactive=False
                            )
                        t2i_last_seed = gr.State(value=-1)

            # ===============================================================
            # TAB 2: Image Edit / Multi-Reference (img2img)
            # ===============================================================
            with gr.TabItem("🖌️ 画像編集・キャラ固定合成 (img2img / 最大8枚参照)", id="tab_edit"):
                with gr.Row():
                    with gr.Column(scale=8):
                        edit_prompt = gr.Textbox(
                            label="編集指示 (Edit Prompt) — ベース画像は <Picture 1>、追加の参照画像は <Picture 2>, <Picture 3>... として指定できます",
                            placeholder='例: <Picture 1> の人物に <Picture 2> の帽子をかぶせて、背景を夕暮れのビーチに変えて。',
                            lines=3,
                        )
                    with gr.Column(scale=3):
                        with gr.Row():
                            edit_gen_btn = gr.Button("🚀 編集を実行 (Generate)", variant="primary", elem_classes=["generate-btn"], scale=3)
                            edit_stop_btn = gr.Button("⏹ 中断", variant="stop", elem_classes=["stop-btn"], scale=1)

                with gr.Row():
                    with gr.Column(scale=5):
                        edit_base_img = gr.Image(
                            type="pil",
                            label="🖼️ 編集するベース画像 (<Picture 1> — 必須)",
                            height=320,
                        )
                        edit_ref_imgs = gr.Gallery(
                            label="➕ 追加の参照画像 (<Picture 2> 〜 <Picture 8> / 任意・最大7枚までドラッグ＆ドロップ可)",
                            columns=4,
                            height=180,
                            interactive=True,
                            file_types=["image"],
                        )

                        gr.Markdown("#### 🔒 ガチ固定モード & 編集方式コントロール")
                        with gr.Row():
                            edit_lock_style = gr.Checkbox(
                                value=True,
                                label="🎨 画風・塗り・線画タッチをガチ固定",
                                info="水彩・色鉛筆・厚塗り・実写などの質感を厳密維持し、勝手な汎用デジタルアニメ塗りに変わるのを防ぎます。",
                            )
                            edit_lock_char = gr.Checkbox(
                                value=True,
                                label="👤 キャラ特徴（髪型形状・目・顔立ち）をガチ固定",
                                info="お団子やリボンの位置、前髪、目の描き方、頭身・体型を <Picture 1> に厳密ロックします。",
                            )

                        edit_denoise = gr.Slider(
                            minimum=0.25,
                            maximum=1.0,
                            step=0.05,
                            value=1.0,
                            label="🎚️ 描き変え強度 / Denoising Strength (1.0 = Image-Edit 構図フリー / 0.50〜0.85 = A1111式 img2img 元構図ベース)",
                            info="ポーズや背景を大きく変える時は 1.0 のまま。元の立ち絵・構図・塗りを下敷きにして服や小物だけ変えたい時は 0.65〜0.85 に下げると劇的に安定します！",
                        )

                        edit_use_upsampler = gr.Checkbox(
                            value=True,
                            label="✨ 画像認識プロンプト拡張 (Vision Upsampler) を使う【推奨】",
                            info="AIが入力画像を実際に見て、一貫性を保つための詳細な編集指示書を自動作成します。",
                        )

                        edit_target_mp = gr.Slider(
                            minimum=1.0,
                            maximum=8.0,
                            step=0.5,
                            value=4.0,
                            label="📐 出力サイズ基準 (Megapixels: 編集時は 4.0 MP 推奨、高精細なら 8.0 MP)",
                        )

                        with gr.Row():
                            edit_seed = gr.Number(value=-1, precision=0, label="🎲 シード値 (-1 でランダム)")
                            edit_random_btn = gr.Button("🎲 -1に戻す", size="sm", scale=0)
                            edit_reuse_btn = gr.Button("♻️ 直前のSeed", size="sm", scale=0)

                        edit_batch = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="連続生成枚数")

                        with gr.Accordion("🔧 詳細設定（プロンプト追従ブースト・手動サイズ・Sigma等）", open=False):
                            edit_ref_boost = gr.Slider(
                                minimum=1.0,
                                maximum=3.0,
                                step=0.1,
                                value=1.0,
                                label="⚡ テキスト編集指示ブースト (DualGuider CFG: 1.0=標準高速 / 1.3〜1.8=指示強制・時間2倍)",
                                info="通常は 1.0 でOK。ガチ固定が強すぎて服や背景が変わりにくい時だけ 1.3〜1.6 に上げてください。",
                            )
                            gr.Markdown("*幅・高さを `0` のままにすると `<Picture 1>` のアスペクト比を自動維持します。*")
                            with gr.Row():
                                edit_custom_w = gr.Slider(0, 4096, step=32, value=0, label="固定幅 (0=自動)")
                                edit_custom_h = gr.Slider(0, 4096, step=32, value=0, label="固定高さ (0=自動)")
                            edit_sampler = gr.Dropdown(
                                choices=["euler", "euler_ancestral", "heun", "dpmpp_2m"],
                                value="euler",
                                label="Sampler",
                            )
                            edit_sigmas = gr.Textbox(value=DEFAULT_SIGMAS, label="Manual Sigmas")
                            with gr.Row():
                                edit_shift_v = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Shift Video")
                                edit_shift_a = gr.Slider(0.1, 5.0, value=1.0, label="Shift Audio")

                    with gr.Column(scale=6):
                        edit_gallery = gr.Gallery(
                            label="編集結果",
                            columns=1,
                            height=620,
                            object_fit="contain",
                            preview=True,
                        )
                        edit_info = gr.Markdown("準備完了")
                        with gr.Row():
                            re_edit_btn = gr.Button("🔁 この編集結果を再びベース画像 (<Picture 1>) にセットして追加編集する", size="sm")
                        with gr.Accordion("📝 AIが作成した編集指示プロンプトを確認", open=False):
                            edit_upsampled_out = gr.Textbox(
                                label="Upsampled Edit Brief", lines=5, interactive=False
                            )
                        edit_last_seed = gr.State(value=-1)

            # ===============================================================
            # TAB 3: History Gallery
            # ===============================================================
            with gr.TabItem("🗂️ 生成履歴ギャラリー (Outputs)", id="tab_history"):
                with gr.Row():
                    hist_refresh_btn = gr.Button("🔄 履歴を最新に更新", variant="secondary")
                hist_gallery = gr.Gallery(
                    value=list_gallery_images(),
                    label="保存済み画像一覧 (outputs/)",
                    columns=5,
                    height=720,
                    object_fit="contain",
                    preview=True,
                )

        gr.Markdown(
            '<div class="footer-badge">Powered by MiniMax H3 &nbsp;|&nbsp; Quantized Weights by Pruna AI (P-H3-Image) &nbsp;|&nbsp; A1111-Style Wrapper UI</div>'
        )

        # -------------------------------------------------------------------
        # Event Wiring
        # -------------------------------------------------------------------
        refresh_btn.click(
            fn=refresh_models,
            outputs=[unet_dd, te_dd, up_dd, vae_dd, status_box],
        )
        free_vram_btn.click(fn=backend.free_vram, outputs=[status_box])
        open_dir_btn.click(fn=open_outputs_folder, outputs=[status_box])

        t2i_preset.change(fn=on_preset_change, inputs=[t2i_preset], outputs=[t2i_width, t2i_height])
        t2i_swap_btn.click(fn=lambda w, h: (h, w), inputs=[t2i_width, t2i_height], outputs=[t2i_width, t2i_height])
        t2i_random_btn.click(fn=lambda: -1, outputs=[t2i_seed])
        t2i_reuse_btn.click(fn=lambda s: s, inputs=[t2i_last_seed], outputs=[t2i_seed])

        t2i_gen_btn.click(
            fn=run_t2i,
            inputs=[
                t2i_prompt,
                t2i_width,
                t2i_height,
                t2i_seed,
                t2i_batch,
                t2i_use_upsampler,
                t2i_sampler,
                t2i_sigmas,
                t2i_shift_v,
                t2i_shift_a,
                unet_dd,
                te_dd,
                up_dd,
                vae_dd,
            ],
            outputs=[t2i_gallery, t2i_upsampled_out, t2i_last_seed, t2i_info, status_box],
        )
        t2i_stop_btn.click(fn=backend.interrupt, outputs=[t2i_info])

        send_to_edit_btn.click(
            fn=send_gallery_to_edit,
            inputs=[t2i_gallery],
            outputs=[edit_base_img, main_tabs],
        )

        edit_random_btn.click(fn=lambda: -1, outputs=[edit_seed])
        edit_reuse_btn.click(fn=lambda s: s, inputs=[edit_last_seed], outputs=[edit_seed])

        edit_gen_btn.click(
            fn=run_edit,
            inputs=[
                edit_prompt,
                edit_base_img,
                edit_ref_imgs,
                edit_lock_style,
                edit_lock_char,
                edit_denoise,
                edit_ref_boost,
                edit_target_mp,
                edit_custom_w,
                edit_custom_h,
                edit_seed,
                edit_batch,
                edit_use_upsampler,
                edit_sampler,
                edit_sigmas,
                edit_shift_v,
                edit_shift_a,
                unet_dd,
                te_dd,
                up_dd,
                vae_dd,
            ],
            outputs=[edit_gallery, edit_upsampled_out, edit_last_seed, edit_info, status_box],
        )
        edit_stop_btn.click(fn=backend.interrupt, outputs=[edit_info])

        re_edit_btn.click(
            fn=send_gallery_to_edit,
            inputs=[edit_gallery],
            outputs=[edit_base_img, main_tabs],
        )

        hist_refresh_btn.click(fn=list_gallery_images, outputs=[hist_gallery])
        main_tabs.select(fn=list_gallery_images, outputs=[hist_gallery])

    return demo


def pick_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="P-H3-Image A1111-style WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio listen host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio listen port")
    parser.add_argument("--comfy-port", type=int, default=8188, help="Internal ComfyUI backend port")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument(
        "--lazy-backend",
        action="store_true",
        help="Wait until first Generate click before starting ComfyUI backend",
    )
    args = parser.parse_args()

    # Check if another ComfyUI is already running on comfy-port; if not, use it
    backend = ComfyBackendManager(host="127.0.0.1", port=args.comfy_port, auto_launch=True)
    if not args.lazy_backend:
        try:
            backend.ensure_running()
        except Exception as e:
            print(f"[WARN] Pre-start of ComfyUI backend failed ({e}); will retry on Generate.")

    ui_port = pick_free_port(args.port)
    demo = create_ui(backend)
    print(f"\n============================================================")
    print(f" P-H3-Image WebUI is starting at: http://{args.host}:{ui_port}")
    print(f"============================================================\n")
    demo.queue().launch(
        server_name=args.host,
        server_port=ui_port,
        inbrowser=not args.no_browser,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=CUSTOM_CSS,
        allowed_paths=[str(OUTPUTS_DIR)],
    )


if __name__ == "__main__":
    main()
