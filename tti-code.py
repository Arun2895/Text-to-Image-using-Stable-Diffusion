from pathlib import Path
import tqdm
from accelerate import Accelerator
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import pandas as pd
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2

class CFG:
  device = "cuda"
  seed = 42
  generator = torch.Generator(device).manual_seed(seed)
  image_gen_steps = 35
  image_gen_model_id = "stabilityai/stable-diffusion-2"
  image_gen_size = (400,400)
  image_gen_guidance_scale = 5
  prompt_gen_model_id = "gpt2"
  prompt_dataset_size = 6
  prompt_max_length = 12

accelerator = Accelerator(cpu=True)
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token="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", guidance_scale=9
)

image_gen_model = accelerator.prepare(image_gen_model)

image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
  image = model(
      prompt,
      num_inference_steps=CFG.image_gen_steps,
      guidance_scale=CFG.image_gen_guidance_scale,
      generator=CFG.generator
  ).images[0]

  image = image.resize(CFG.image_gen_size)
  return image

generate_image("a latest mustang car", image_gen_model)
