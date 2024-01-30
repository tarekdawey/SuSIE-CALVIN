Befor using retrieve_text_vector.py script, add the following function at the end of the code in model.py script in the cloned susie folder:
def get_latent_vector_of_prompt(prompt, tokenize, text_encode):
    prompt_ids = tokenize([prompt])
    return text_encode(prompt_ids)
