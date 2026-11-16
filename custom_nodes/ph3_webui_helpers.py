"""Custom helper nodes for P-H3-Image WebUI:
1. PH3StrictBriefPostprocess: Enforces exact art style, linework, texture, and character
   trait preservation on the Qwen3.5 upsampler output (prevents generic "Anime illustration"
   drift when Lock Style / Character mode is enabled).
2. PH3ImageToAVLatent: Enables classic A1111-style img2img with controllable Denoising
   Strength (0.05 - 1.0) on P-H3-Image AV latents while respecting PDD head sigma intervals.
"""

import re
import torch
import comfy.model_management
import comfy.nested_tensor


class PH3StrictBriefPostprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brief": ("STRING", {"forceInput": True}),
                "lock_style": ("BOOLEAN", {"default": True}),
                "lock_character": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("brief",)
    FUNCTION = "process"
    CATEGORY = "text/minimax"

    def process(self, brief: str, lock_style: bool = True, lock_character: bool = True):
        out = brief or ""
        if not (lock_style or lock_character):
            return (out,)

        # Upgrade partially_preserved -> fully_preserved in retention_analysis when locked
        if lock_style and lock_character:
            out = re.sub(
                r"(<Subject 1>:\s*)partially_preserved",
                r"\1fully_preserved",
                out,
            )

        # Strip genericizing "Anime illustration style." prefix if the source wasn't explicitly asked to become generic anime
        reinforcements = []
        if lock_style:
            reinforcements.append(
                "Strictly replicate the exact rendering medium, linework weight, brushwork/coloring texture, shading softness, and color palette of <Picture 1>; do NOT convert it into generic glossy digital anime shading."
            )
        if lock_character:
            reinforcements.append(
                "Keep <Subject 1>'s exact facial proportions, eye shape and iris rendering, exact hair silhouette/tied style/accessories, and body proportions identical to <Picture 1>."
            )

        addon = " ".join(reinforcements)
        if "detailed_description:" in out:
            out = out.rstrip() + "\n" + addon
        else:
            out = out.rstrip() + " " + addon

        return (out,)


class PH3ImageToAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "denoise": ("FLOAT", {"default": 0.70, "min": 0.10, "max": 1.0, "step": 0.05}),
                "sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("LATENT", "SIGMAS")
    RETURN_NAMES = ("latent", "sigmas")
    FUNCTION = "encode_img2img"
    CATEGORY = "model/latent/minimax"

    def encode_img2img(self, vae, image, denoise, sigmas):
        device = comfy.model_management.intermediate_device()
        b = 1
        z_audio = torch.zeros([b, 32, 2, 2], device=device, dtype=torch.float32)

        if denoise >= 0.999:
            h, w = image.shape[1], image.shape[2]
            empty_v = torch.zeros([b, 24, 1, h // 16, w // 16], device=device, dtype=torch.float32)
            return ({"samples": comfy.nested_tensor.NestedTensor((empty_v, z_audio))}, sigmas)

        # Encode clean source image -> [1, 24, 1, H/16, W/16]
        z_video = vae.encode(image[:1, ..., :3]).to(device=device, dtype=torch.float32)

        # P-H3-Image uses an 8-interval PDD head bank tied to its 9-point sigma schedule:
        # [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]
        # Slicing directly onto existing grid points guarantees 100% alignment with PDD heads!
        total_steps = max(len(sigmas) - 1, 1)
        active_steps = max(1, min(total_steps, round(total_steps * float(denoise))))
        sliced_sigmas = sigmas[-(active_steps + 1):].clone()

        # ComfyUI's CONST.noise_scaling automatically mixes:
        #   x_t = start_sigma * noise + (1.0 - start_sigma) * z_video
        return ({"samples": comfy.nested_tensor.NestedTensor((z_video, z_audio))}, sliced_sigmas)


NODE_CLASS_MAPPINGS = {
    "PH3StrictBriefPostprocess": PH3StrictBriefPostprocess,
    "PH3ImageToAVLatent": PH3ImageToAVLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PH3StrictBriefPostprocess": "P-H3 Strict Brief Postprocess",
    "PH3ImageToAVLatent": "P-H3 Image to AV Latent (img2img)",
}
