import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence, Union

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
import uvicorn


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Access helper that gracefully handles mapping outputs from nodes."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str | None = None) -> str | None:
    """Walk up the directory tree until *name* is found."""
    current_path = path or os.getcwd()
    if name in os.listdir(current_path):
        located = os.path.join(current_path, name)
        print(f"{name} found: {located}")
        return located
    parent = os.path.dirname(current_path)
    if parent == current_path:
        return None
    return find_path(name, parent)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path and os.path.isdir(comfyui_path) and comfyui_path not in sys.path:
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        print("Falling back to utils.extra_config.load_extra_path_config")
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

import folder_paths


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_extra_nodes

    sys_path_entry = find_path("ComfyUI")
    if sys_path_entry and sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)

    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())


from nodes import (
    CLIPLoader,
    CLIPTextEncode,
    KSamplerAdvanced,
    LoadImage,
    NODE_CLASS_MAPPINGS,
    VAEDecode,
    VAELoader,
    LoraLoaderModelOnly,
)


DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, image, still, "
    "overall grayish, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, distorted limbs, fingers fused together, "
    "static image, cluttered background, three legs, many people in the background, walking backwards."
)

CLIP_MODEL_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_MODEL_NAME = "wan_2.1_vae.safetensors"
HIGH_NOISE_UNET = "wan2.2_i2v_high_noise_14B_Q4_K_S.gguf"
LOW_NOISE_UNET = "wan2.2_i2v_low_noise_14B_Q4_K_S.gguf"
HIGH_NOISE_LORA = "high_noise_model.safetensors"
LOW_NOISE_LORA = "low_noise_model.safetensors"
SAGE_ATTENTION_MODE = "sageattn_qk_int8_pv_fp8_cuda++"


class WanVideoGenerator:
    """Wraps the Wan 2.2 workflow and caches heavy models between runs."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._nodes_ready = False
        self._clip_cache: Any | None = None
        self._vae_cache: Any | None = None
        self._high_noise_model: Any | None = None
        self._low_noise_model: Any | None = None

    def initialize(self) -> None:
        if self._nodes_ready:
            return
        import_custom_nodes()
        self.cliploader = CLIPLoader()
        self.cliptextencode = CLIPTextEncode()
        self.vaeloader = VAELoader()
        self.unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        self.patch_sage = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
        self.torch_settings = NODE_CLASS_MAPPINGS["ModelPatchTorchSettings"]()
        self.lora_loader = LoraLoaderModelOnly()
        self.modelsampling = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        self.imageresize = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        self.wanimagetovideo = NODE_CLASS_MAPPINGS["WanImageToVideo"]()
        self.ksampler = KSamplerAdvanced()
        self.vaedecode = VAEDecode()
        self.vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        self.loadimage = LoadImage()
        self._nodes_ready = True

    def _ensure_model_cache(self) -> None:
        if not self._nodes_ready:
            raise RuntimeError("Generator not initialized. Call initialize() first.")

        if self._clip_cache is None:
            self._clip_cache = self.cliploader.load_clip(
                clip_name=CLIP_MODEL_NAME, type="wan", device="cpu"
            )

        if self._vae_cache is None:
            self._vae_cache = self.vaeloader.load_vae(vae_name=VAE_MODEL_NAME)

        if self._high_noise_model is None:
            self._high_noise_model = self._build_lora_model(
                HIGH_NOISE_UNET, HIGH_NOISE_LORA, 3.0
            )

        if self._low_noise_model is None:
            self._low_noise_model = self._build_lora_model(
                LOW_NOISE_UNET, LOW_NOISE_LORA, 1.5
            )

    def _build_lora_model(self, unet_name: str, lora_name: str, strength: float) -> Any:
        unet = self.unet_loader.load_unet(unet_name=unet_name)
        patched = self.patch_sage.patch(
            sage_attention=SAGE_ATTENTION_MODE,
            model=get_value_at_index(unet, 0),
        )
        torch_ready = self.torch_settings.patch(
            enable_fp16_accumulation=True,
            model=get_value_at_index(patched, 0),
        )
        return self.lora_loader.load_lora_model_only(
            lora_name=lora_name,
            strength_model=strength,
            model=get_value_at_index(torch_ready, 0),
        )

    def generate_video(self, prompt: str, negative_prompt: str, image_name: str) -> dict[str, Any]:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt cannot be empty")
        cleaned_negative = (negative_prompt or DEFAULT_NEGATIVE_PROMPT).strip()
        if not cleaned_negative:
            cleaned_negative = DEFAULT_NEGATIVE_PROMPT

        with self._lock:
            self._ensure_model_cache()
            return self._run_generation(cleaned_prompt, cleaned_negative, image_name)

    def _run_generation(self, prompt: str, negative_prompt: str, image_name: str) -> dict[str, Any]:
        with torch.inference_mode():
            positive_conditioning = self.cliptextencode.encode(
                text=prompt,
                clip=get_value_at_index(self._clip_cache, 0),
            )
            negative_conditioning = self.cliptextencode.encode(
                text=negative_prompt,
                clip=get_value_at_index(self._clip_cache, 0),
            )

            loaded_image = self.loadimage.load_image(image=image_name)
            resize_result = self.imageresize.resize(
                width=480,
                height=480,
                upscale_method="lanczos",
                keep_proportion="crop",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=16,
                device="cpu",
                image=get_value_at_index(loaded_image, 0),
                unique_id=random.randint(1, 2**63 - 1),
            )

            low_noise_sampling = self.modelsampling.patch(
                shift=8.0,
                model=get_value_at_index(self._low_noise_model, 0),
            )
            high_noise_sampling = self.modelsampling.patch(
                shift=8.0,
                model=get_value_at_index(self._high_noise_model, 0),
            )

            wanimagetovideo_result = self.wanimagetovideo.EXECUTE_NORMALIZED(
                width=get_value_at_index(resize_result, 1),
                height=get_value_at_index(resize_result, 2),
                length=81,
                batch_size=1,
                positive=get_value_at_index(positive_conditioning, 0),
                negative=get_value_at_index(negative_conditioning, 0),
                vae=get_value_at_index(self._vae_cache, 0),
                start_image=get_value_at_index(resize_result, 0),
            )

            ksampler_first = self.ksampler.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64 - 1),
                steps=6,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                start_at_step=0,
                end_at_step=3,
                return_with_leftover_noise="enable",
                model=get_value_at_index(high_noise_sampling, 0),
                positive=get_value_at_index(wanimagetovideo_result, 0),
                negative=get_value_at_index(wanimagetovideo_result, 1),
                latent_image=get_value_at_index(wanimagetovideo_result, 2),
            )

            ksampler_second = self.ksampler.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64 - 1),
                steps=6,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                start_at_step=3,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(low_noise_sampling, 0),
                positive=get_value_at_index(wanimagetovideo_result, 0),
                negative=get_value_at_index(wanimagetovideo_result, 1),
                latent_image=get_value_at_index(ksampler_first, 0),
            )

            decoded_frames = self.vaedecode.decode(
                samples=get_value_at_index(ksampler_second, 0),
                vae=get_value_at_index(self._vae_cache, 0),
            )

            date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
            video_meta = self.vhs_videocombine.combine_video(
                frame_rate=16,
                loop_count=0,
                filename_prefix=f"{date_prefix}/wan22_",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(decoded_frames, 0),
                unique_id=random.randint(1, 2**63 - 1),
            )

        preview = video_meta.get("ui", {}).get("gifs", [None])[0]
        result_tuple = video_meta.get("result", ())
        output_files = []
        if result_tuple:
            _save_flag, output_files = result_tuple[0]

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "frame_rate": preview.get("frame_rate") if preview else 16,
            "output_path": preview.get("fullpath") if preview else (output_files[-1] if output_files else None),
            "output_subfolder": preview.get("subfolder") if preview else None,
            "output_filename": preview.get("filename") if preview else None,
            "output_files": output_files,
        }


generator = WanVideoGenerator()
app = FastAPI(
    title="Wan22 Video Generation",
    version="0.1.0",
    description="FastAPI wrapper around the Wan 2.2 ComfyUI workflow with cached models.",
)


async def _persist_upload(image: UploadFile) -> tuple[str, str]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    safe_name = Path(image.filename or "input.png").name
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    destination = os.path.join(input_dir, unique_name)
    with open(destination, "wb") as output_file:
        output_file.write(data)
    return unique_name, destination


def _cleanup_file(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


@app.on_event("startup")
async def startup_event() -> None:
    await run_in_threadpool(generator.initialize)


@app.post("/generate")
async def generate_endpoint(
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    image: UploadFile = File(...),
):
    image_name, image_path = await _persist_upload(image)
    try:
        result = await run_in_threadpool(
            generator.generate_video,
            prompt,
            negative_prompt,
            image_name,
        )
    except Exception as exc:  # pragma: no cover - surfaced via HTTPException
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup_file(image_path)

    return result


def main() -> None:
    port = int(os.environ.get("FASTAPI_PORT", "8000"))
    reload_enabled = os.environ.get("FASTAPI_RELOAD", "0") == "1"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
