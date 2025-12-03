import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes

    sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


from nodes import (
    CLIPTextEncode,
    VAELoader,
    KSamplerAdvanced,
    VAEDecode,
    NODE_CLASS_MAPPINGS,
    LoadImage,
    LoraLoaderModelOnly,
    CLIPLoader,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        cliploader = CLIPLoader()
        cliploader_38 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="cpu"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="a video where the camera moves in a physically realistic manner, towards the tree. the camera moves rapidly, reaching the tree within 3 seconds",
            clip=get_value_at_index(cliploader_38, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, painting, image, still, overall grayish, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, distorted limbs, fingers fused together, static image, cluttered background, three legs, many people in the background, walking backwards.",
            clip=get_value_at_index(cliploader_38, 0),
        )

        vaeloader = VAELoader()
        vaeloader_39 = vaeloader.load_vae(vae_name="wan_2.1_vae.safetensors")

        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unetloadergguf_61 = unetloadergguf.load_unet(
            unet_name="wan2.2_i2v_high_noise_14B_Q4_K_S.gguf"
        )

        unetloadergguf_62 = unetloadergguf.load_unet(
            unet_name="wan2.2_i2v_low_noise_14B_Q4_K_S.gguf"
        )

        pathchsageattentionkj = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
        pathchsageattentionkj_65 = pathchsageattentionkj.patch(
            sage_attention="sageattn_qk_int8_pv_fp8_cuda++",
            model=get_value_at_index(unetloadergguf_61, 0),
        )

        modelpatchtorchsettings = NODE_CLASS_MAPPINGS["ModelPatchTorchSettings"]()
        modelpatchtorchsettings_66 = modelpatchtorchsettings.patch(
            enable_fp16_accumulation=True,
            model=get_value_at_index(pathchsageattentionkj_65, 0),
        )

        loraloadermodelonly = LoraLoaderModelOnly()
        loraloadermodelonly_69 = loraloadermodelonly.load_lora_model_only(
            lora_name="high_noise_model.safetensors",
            strength_model=3.0000000000000004,
            model=get_value_at_index(modelpatchtorchsettings_66, 0),
        )

        pathchsageattentionkj_67 = pathchsageattentionkj.patch(
            sage_attention="sageattn_qk_int8_pv_fp8_cuda++",
            model=get_value_at_index(unetloadergguf_62, 0),
        )

        modelpatchtorchsettings_68 = modelpatchtorchsettings.patch(
            enable_fp16_accumulation=True,
            model=get_value_at_index(pathchsageattentionkj_67, 0),
        )

        loraloadermodelonly_70 = loraloadermodelonly.load_lora_model_only(
            lora_name="low_noise_model.safetensors",
            strength_model=1.5000000000000002,
            model=get_value_at_index(modelpatchtorchsettings_68, 0),
        )

        loadimage = LoadImage()
        loadimage_80 = loadimage.load_image(image="unnamed.jpg")

        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        wanimagetovideo = NODE_CLASS_MAPPINGS["WanImageToVideo"]()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            modelsamplingsd3_55 = modelsamplingsd3.patch(
                shift=8.000000000000002,
                model=get_value_at_index(loraloadermodelonly_70, 0),
            )

            imageresizekjv2_64 = imageresizekjv2.resize(
                width=480,
                height=480,
                upscale_method="lanczos",
                keep_proportion="crop",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=16,
                device="cpu",
                image=get_value_at_index(loadimage_80, 0),
                unique_id=4788502543572388243,
            )

            wanimagetovideo_50 = wanimagetovideo.EXECUTE_NORMALIZED(
                width=get_value_at_index(imageresizekjv2_64, 1),
                height=get_value_at_index(imageresizekjv2_64, 2),
                length=81,
                batch_size=1,
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                vae=get_value_at_index(vaeloader_39, 0),
                start_image=get_value_at_index(imageresizekjv2_64, 0),
            )

            modelsamplingsd3_54 = modelsamplingsd3.patch(
                shift=8.000000000000002,
                model=get_value_at_index(loraloadermodelonly_69, 0),
            )

            ksampleradvanced_57 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=6,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                start_at_step=0,
                end_at_step=3,
                return_with_leftover_noise="enable",
                model=get_value_at_index(modelsamplingsd3_54, 0),
                positive=get_value_at_index(wanimagetovideo_50, 0),
                negative=get_value_at_index(wanimagetovideo_50, 1),
                latent_image=get_value_at_index(wanimagetovideo_50, 2),
            )

            ksampleradvanced_58 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=6,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                start_at_step=3,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(modelsamplingsd3_55, 0),
                positive=get_value_at_index(wanimagetovideo_50, 0),
                negative=get_value_at_index(wanimagetovideo_50, 1),
                latent_image=get_value_at_index(ksampleradvanced_57, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_58, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            vhs_videocombine_63 = vhs_videocombine.combine_video(
                frame_rate=16,
                loop_count=0,
                filename_prefix="2025-12-02/wan22_",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(vaedecode_8, 0),
                unique_id=17304478183971321280,
            )


if __name__ == "__main__":
    main()
