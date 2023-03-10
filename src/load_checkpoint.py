from diffusers import StableDiffusionPipeline
import torch

model_path = "/home/ray/default/fine-tune-stable-diffusion/src/sd-model-finetuned"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("yoda-pokemon.png")
