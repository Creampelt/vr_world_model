import asyncio
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence, Union
from tqdm import tqdm, trange

from fastapi.concurrency import run_in_threadpool
import torch
import traceback
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from contextlib import asynccontextmanager
import uvicorn
from pydantic import BaseModel, Field, model_validator

# =========================================================================
# ============================= COMFY HELPERS =============================
# =========================================================================


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


# ============================= COMFY SETUP =============================

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


# =========================================================================
# ========================== WAN VIDEO API IMPLEMENTATION =================
# =========================================================================

# ========================== NODES REQUIRED ===============================

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


CLIP_MODEL_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_MODEL_NAME = "wan_2.1_vae.safetensors"
HIGH_NOISE_UNET = "wan2.2_i2v_high_noise_14B_Q4_K_S.gguf"
LOW_NOISE_UNET = "wan2.2_i2v_low_noise_14B_Q4_K_S.gguf"
HIGH_NOISE_LORA = "high_noise_model.safetensors"
LOW_NOISE_LORA = "low_noise_model.safetensors"
SAGE_ATTENTION_MODE = "sageattn_qk_int8_pv_fp8_cuda++"


DEFAULT_NEGATIVE_PROMPT = """
Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, image, still,
overall grayish, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers,
poorly drawn hands, poorly drawn faces, deformed, disfigured, distorted limbs, fingers fused together,
static image, cluttered background, three legs, many people in the background, walking backwards.
"""
DEFAULT_NEGATIVE_PROMPT = " ".join(
    line.strip() for line in DEFAULT_NEGATIVE_PROMPT.strip().splitlines()
)
NUM_DENOSING_STEPS = 6


class WanVideoGenerator:
    """Wraps the Wan 2.2 workflow and caches heavy models between runs."""

    def __init__(
        self,
        denoising_steps: int,
    ) -> None:
        self._lock = Lock()
        self._nodes_ready = False
        self._denoising_steps = denoising_steps

        if self._denoising_steps % 2 != 0 or self._denoising_steps < 2:
            raise ValueError("denoising_steps must be an even number at least 2")

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

    def generate_videos(
        self,
        images: list[str],
        positive_prompts: list[str],
        negative_prompts: list[str],
    ) -> list[dict[str, Any]]:
        positive_prompts = [p.strip() for p in positive_prompts]
        negative_prompts = [p.strip() for p in negative_prompts]
        image_buffers = [self.loadimage.load_image(image)[0] for image in images]
        image_buffers = torch.cat(image_buffers, dim=0)

        with self._lock:
            return self._run_generation(
                image_buffers,
                positive_prompts,
                negative_prompts,
            )

    def _build_lora_model(
        self,
        unet_name: str,
        lora_name: str,
        strength: float,
    ) -> Any:
        unet = self.unet_loader.load_unet(unet_name=unet_name)
        patched = self.patch_sage.patch(
            sage_attention=SAGE_ATTENTION_MODE,
            model=get_value_at_index(unet, 0),
        )
        torch_ready = self.torch_settings.patch(
            enable_fp16_accumulation=True,
            model=get_value_at_index(patched, 0),
        )
        lora_loaded = self.lora_loader.load_lora_model_only(
            lora_name=lora_name,
            strength_model=strength,
            model=get_value_at_index(torch_ready, 0),
        )
        return lora_loaded

    def _run_generation(
        self,
        image_buffers: torch.Tensor,
        positive_prompts: list[str],
        negative_prompts: list[str],
        n_frames: int = 81,
    ) -> dict[str, Any]:
        batch_size = image_buffers.shape[0]

        with torch.inference_mode():
            # load clip and encode prompts
            print("Loading CLIP model")
            clip_model = get_value_at_index(
                self.cliploader.load_clip(
                    clip_name=CLIP_MODEL_NAME,
                    type="wan",
                    device="cpu",
                ),
                0,
            )
            positive_conditioning = []
            for prompt in tqdm(positive_prompts, desc="Encoding Positive Prompts"):
                encoded = self.cliptextencode.encode(
                    text=prompt,
                    clip=clip_model,
                )
                positive_conditioning.append(get_value_at_index(encoded, 0))
            negative_conditioning = []
            for prompt in tqdm(negative_prompts, desc="Encoding Negative Prompts"):
                encoded = self.cliptextencode.encode(
                    text=prompt,
                    clip=clip_model,
                )
                negative_conditioning.append(get_value_at_index(encoded, 0))

            # resize the images and prepare for video generation
            print("Resizing input images")
            resize_result = self.imageresize.resize(
                width=480,
                height=480,
                upscale_method="lanczos",
                keep_proportion="crop",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=16,
                device="cpu",
                image=image_buffers,
                unique_id=100,
            )

            # load the vae
            print("Loading VAE model")
            vae = get_value_at_index(
                self.vaeloader.load_vae(vae_name=VAE_MODEL_NAME),
                0,
            )

            # setup wan image to video
            wanimagetovideo_results = []
            for i in tqdm(range(batch_size), desc="WanImageToVideo Generation"):
                start_image = get_value_at_index(resize_result, 0)[i : i + 1]
                result = self.wanimagetovideo.EXECUTE_NORMALIZED(
                    width=get_value_at_index(resize_result, 1),
                    height=get_value_at_index(resize_result, 2),
                    length=n_frames,
                    batch_size=1,
                    positive=positive_conditioning[i],
                    negative=negative_conditioning[i],
                    vae=vae,
                    start_image=start_image,
                )
                wanimagetovideo_results.append(result)

            # high noise sampling
            high_noise_model = get_value_at_index(
                self._build_lora_model(
                    HIGH_NOISE_UNET,
                    HIGH_NOISE_LORA,
                    3.0,
                ),
                0,
            )
            high_noise_sampling = self.modelsampling.patch(
                shift=8.0,
                model=high_noise_model,
            )
            high_noise_results = []
            for i in tqdm(range(batch_size), desc="High Noise Sampling"):
                wani_result = wanimagetovideo_results[i]
                ksampler_high = self.ksampler.sample(
                    add_noise="enable",
                    noise_seed=random.randint(1, 2**64 - 1),
                    steps=self._denoising_steps,
                    cfg=1,
                    sampler_name="euler",
                    scheduler="simple",
                    start_at_step=0,
                    end_at_step=self._denoising_steps // 2,
                    return_with_leftover_noise="enable",
                    model=get_value_at_index(high_noise_sampling, 0),
                    positive=get_value_at_index(wani_result, 0),
                    negative=get_value_at_index(wani_result, 1),
                    latent_image=get_value_at_index(wani_result, 2),
                )
                high_noise_results.append(ksampler_high)

            # low noise sampling
            low_noise_model = get_value_at_index(
                self._build_lora_model(
                    LOW_NOISE_UNET,
                    LOW_NOISE_LORA,
                    1.5,
                ),
                0,
            )
            low_noise_sampling = self.modelsampling.patch(
                shift=8.0,
                model=low_noise_model,
            )
            low_noise_results = []
            for i in tqdm(range(batch_size), desc="Low Noise Sampling"):
                wani_result = wanimagetovideo_results[i]
                ksampler_high = high_noise_results[i]
                ksampler_low = self.ksampler.sample(
                    add_noise="disable",
                    noise_seed=random.randint(1, 2**64 - 1),
                    steps=self._denoising_steps,
                    cfg=1,
                    sampler_name="euler",
                    scheduler="simple",
                    start_at_step=self._denoising_steps // 2,
                    end_at_step=self._denoising_steps,
                    return_with_leftover_noise="disable",
                    model=get_value_at_index(low_noise_sampling, 0),
                    positive=get_value_at_index(wani_result, 0),
                    negative=get_value_at_index(wani_result, 1),
                    latent_image=get_value_at_index(ksampler_high, 0),
                )
                low_noise_results.append(ksampler_low)

            # decode and combine videos
            results = []
            for i in tqdm(range(batch_size), desc="Decoding and Combining Videos"):
                ksampler_second = low_noise_results[i]
                decoded_frames = self.vaedecode.decode(
                    samples=get_value_at_index(ksampler_second, 0),
                    vae=vae,
                )
                date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

                results.append(
                    {
                        "positive_prompt": positive_prompts[i],
                        "negative_prompt": negative_prompts[i],
                        "frame_rate": preview.get("frame_rate") if preview else 16,
                        "output_path": preview.get("fullpath")
                        if preview
                        else (output_files[-1] if output_files else None),
                        "output_subfolder": preview.get("subfolder") if preview else None,
                        "output_filename": preview.get("filename") if preview else None,
                        "output_files": output_files,
                    }
                )

        return results


generator = WanVideoGenerator(denoising_steps=NUM_DENOSING_STEPS)

# ========================== API SETUP HELPERS ===============================


async def _persist_upload(image: UploadFile, index_id: int) -> tuple[str, str]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    safe_name = Path(image.filename or "input.png").name
    safe_stem = Path(safe_name).stem
    safe_ext = Path(safe_name).suffix
    unique_name = f"{safe_stem}_{index_id}_{uuid.uuid4().hex}{safe_ext}"
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    destination = os.path.join(input_dir, unique_name)
    with open(destination, "wb") as output_file:
        output_file.write(data)
    return unique_name, destination


def _cleanup_file(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


# =========================================================================
# ============================== API ENDPOINTS ============================
# =========================================================================

# ================================ LIFESPAN ================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(generator.initialize)
    yield


app = FastAPI(
    title="Wan22 Video Generation",
    version="0.1.0",
    description="FastAPI wrapper around the Wan 2.2 ComfyUI workflow with cached models.",
    lifespan=lifespan,
)

# ================================ GENERATE ================================


class GeneratePrompts(BaseModel):
    positive_prompts: list[str]
    negative_prompts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_lengths_and_fill_negatives(self):
        if not self.negative_prompts:
            self.negative_prompts = [DEFAULT_NEGATIVE_PROMPT] * len(
                self.positive_prompts
            )
        if len(self.positive_prompts) != len(self.negative_prompts):
            raise ValueError(
                "Length of positive_prompts and negative_prompts must match."
            )
        return self


@app.post("/generate")
async def generate_endpoint(
    prompts_json: str = Form(...),
    images: list[UploadFile] = File(...),
):
    # check inputs
    prompts = GeneratePrompts.model_validate_json(prompts_json)
    if len(images) != len(prompts.positive_prompts):
        raise HTTPException(
            status_code=400,
            detail="Number of uploaded images must match number of positive prompts.",
        )

    # execute generation in batch
    # persist all uploads asynchronously
    uploaded_files = await asyncio.gather(
        *[_persist_upload(image, i) for i, image in enumerate(images)]
    )
    image_names = [name for name, _ in uploaded_files]
    image_paths = [path for _, path in uploaded_files]
    try:
        result = await run_in_threadpool(
            generator.generate_videos,
            image_names,
            prompts.positive_prompts,
            prompts.negative_prompts,
        )
    except Exception as exc:  # pragma: no cover - surfaced via HTTPException
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for image_path in image_paths:
            _cleanup_file(image_path)

    return result


# =========================================================================
# ================================ MAIN ===================================
# =========================================================================


def main() -> None:
    port = int(os.environ.get("FASTAPI_PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
