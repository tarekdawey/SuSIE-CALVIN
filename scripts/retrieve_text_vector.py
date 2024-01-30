from susie.model import create_sample_fn, get_latent_vector_of_prompt, load_text_encoder
from susie.jax_utils import initialize_compilation_cache
import numpy as np
import os

# Initialize compilation cache
initialize_compilation_cache()

# Load the text encoder functions
tokenize, _, text_encode = load_text_encoder("runwayml/stable-diffusion-v1-5:flax")

# Create the sample function
sample_fn = create_sample_fn(
    os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
    "kvablack/dlimp-diffusion/9n9ped8m",
    num_timesteps=200,
    prompt_w=7.5,
    context_w=1.5,
    eta=0.0,
    pretrained_path="runwayml/stable-diffusion-v1-5:flax",
)

def encode_language(language_command, tokenize, text_encode):
    latent_vector_jax = get_latent_vector_of_prompt(language_command, tokenize, text_encode)
    return np.asarray(latent_vector_jax)

def save_latent_vector(latent_vector, directory='/path/to/save/new_vector', file_name='rotate_red_block_left.npy'):
    if latent_vector is not None:
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, file_name)
        np.save(save_path, latent_vector)
    else:
        print("Latent vector is not set.")

# Example usage
language_command = "take the red block and rotate it to the left"
latent_vector = encode_language(language_command, tokenize, text_encode)
save_latent_vector(latent_vector)
