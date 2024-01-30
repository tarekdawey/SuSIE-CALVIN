import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from adjustText import adjust_text

def load_vectors(directory):
    file_names = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
    vectors = [np.load(file_name).reshape(-1) for file_name in file_names]  # Reshape each vector
    return np.array(vectors)

# Load vectors
directory = '/home/systemtec/calvin-sim/latent tasks/vectors'
vectors = load_vectors(directory)


# Load your language prompts here
# For example, prompts = ['prompt1', 'prompt2', ..., 'prompt34']
prompts = ['rotate_red_block_right', 'rotate_red_block_left', 'rotate_blue_block_right', 'rotate_blue_block_left', 'rotate_pink_block_right', 'rotate_pink_block_left', 'push_red_block_right', 'push_red_block_left', 'push_blue_block_right', 'push_blue_block_left', 'push_pink_block_right', 'push_pink_block_left', 'move_slider_left', 'move_slider_right', 'open_drawer', 'close_drawer', 'lift_red_block_table', 'lift_blue_block_table', 'lift_pink_block_table', 'lift_red_block_slider', 'lift_blue_block_slider', 'lift_pink_block_slider', 'lift_red_block_drawer', 'lift_blue_block_drawer', 'lift_pink_block_drawer', 'place_in_slider', 'place_in_drawer', 'push_into_drawer', 'stack_block', 'unstack_block', 'turn_on_lightbulb', 'turn_off_lightbulb', 'turn_on_led', 'turn_off_led']  # Replace with your actual prompts

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
vectors_tsne = tsne.fit_transform(vectors)

# t-SNE plot with arrows
plt.figure(figsize=(18, 15))
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
texts = [plt.text(vectors_tsne[i, 0], vectors_tsne[i, 1], prompt, fontsize=8, ha='right') for i, prompt in enumerate(prompts)]
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='red'))
plt.title('t-SNE Visualization')
plt.savefig('/home/systemtec/calvin-sim/tsne_visualization.png')
plt.close()

# Apply UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
vectors_umap = umap_model.fit_transform(vectors)

# UMAP plot
plt.figure(figsize=(12, 10))
plt.scatter(vectors_umap[:, 0], vectors_umap[:, 1])
texts = [plt.text(vectors_umap[i, 0], vectors_umap[i, 1], prompt, fontsize=8) for i, prompt in enumerate(prompts)]
adjust_text(texts)
plt.title('UMAP Visualization')
plt.savefig('/home/systemtec/calvin-sim/umap_visualization.png')
plt.close()

