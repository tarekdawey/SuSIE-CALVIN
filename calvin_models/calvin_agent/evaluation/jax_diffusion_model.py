from denoising_diffusion_flax.model import create_sample_fn
from denoising_diffusion_flax.jax_utils import initialize_compilation_cache
import numpy as np
from PIL import Image
import os

class DiffusionModel:
    def __init__(self):
        initialize_compilation_cache()

        self.sample_fn = create_sample_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            num_timesteps=200,
            prompt_w=7.5,
            context_w=1.5,
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5:flax",
        )

    def generate(self, language_command : str, image_obs : np.ndarray):
        # Resize image to 256x256
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)

        sample = self.sample_fn(image_obs, language_command, prompt_w=7.5, context_w=1.5)
        return np.array(Image.fromarray(sample).resize((200, 200))).astype(np.uint8)