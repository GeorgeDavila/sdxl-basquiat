# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import sys
import torch
import shutil
from PIL import Image
from typing import List
from diffusers import StableDiffusionXLPipeline

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
LoRA_PATH = "GDavila/sdxl-basquiat"
LoRA_file = "lora.safetensors"
device = "cuda"
MODEL_CACHE = "model-cache"

def load_image(path): #for img2img
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # load SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16,
            use_safetensors=True,
            watermark=None,
            safety_checker=None,
            variant="fp16",
        ).to(device)

        self.pipe.load_lora_weights(LoRA_PATH, weight_name=LoRA_file)

    def predict(
        self,
        myprompt: str = Input(
            description="Input prompt (BASQUIAT is the style trigger word)",
            default="A BASQUIAT painting"
        ),
        promptAddendum: str = Input(
            description="Extra terms to add to end of prompt",
            default="",
        ),
        negative_prompt: str = Input(
            description="Negative Prompt",
            default=""
        ),
        outWidth: int = Input(
            description="width of output",
            ge=128,
            le=4096,
            default=1024,
        ),
        outHeight: int = Input(
            description="height of output",
            ge=128,
            le=4096,
            default=1024,
        ),
        guidanceScale: float = Input(
            description="Guidance scale (influence of input text on generation)",
            ge=0.0,
            le=50.0,
            default=7.5
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        while (myprompt[-1] == " ") or (myprompt[-1] == "\n"): #remove user whitespaces
            myprompt = myprompt[:-1]
        myprompt = myprompt + promptAddendum
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        imagesObj = self.pipe(
            prompt=myprompt, 
            height=outHeight, #default 1024
            width=outWidth, #default 1024
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidanceScale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_outputs,
            seed=seed,
            lora_scale=lora_scale,
            ).images #[0] #.images[0]
        
        output_paths = []
        for i, _ in enumerate(imagesObj):
            output_path = f"/tmp/out-{i}.png"
            imagesObj[i].save(output_path)
            output_paths.append(Path(output_path))
            
        return output_paths
