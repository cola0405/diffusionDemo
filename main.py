# pip install --upgrade diffusers[torch]==0.12.1
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

prompt = "white horse in universe"

image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")