import numpy as np
import os
import random

# Directory containing CALVIN .npy files
directory = '/path/to/calvin/vectors'

# Load vectors
file_names = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
latent_vectors = [np.load(file_name).reshape(77, 768) for file_name in file_names]  # Reshape each vector to (77, 768)

# Initialize arrays to store min, max, and mean values
min_values = np.zeros((77, 768))
max_values = np.zeros((77, 768))
mean_values = np.zeros((77, 768))

# Extract min, max, and mean values for each dimension
for i in range(77):
    for j in range(768):
        dimension_values = [vec[i, j] for vec in latent_vectors]
        min_values[i, j] = min(dimension_values)
        max_values[i, j] = max(dimension_values)
        mean_values[i, j] = np.mean(dimension_values)


# Generate New Latent Vectors by assigned either the minimum or maximum value
# Generate 100 new latent vectors
new_latent_vectors = []
for _ in range(100):
    new_vector = np.zeros((77, 768))
    for i in range(77):
        for j in range(768):
            new_vector[i, j] = random.choice([min_values[i, j], max_values[i, j]])
    new_latent_vectors.append(new_vector.reshape(1, 77, 768))

# Optionally, save these vectors as .npy files
for idx, vec in enumerate(new_latent_vectors):
    np.save(f'/path/to/save/vectors/random_min_max/new_vector_{idx}.npy', vec)


# Generate New Latent Vectors by assigned Values Within Min-Max Range
def random_value_in_range(min_val, max_val):
    return np.random.uniform(min_val, max_val)

# Generate 100 new latent vectors with random values in the min-max range
new_latent_vectors_random_range = []
for _ in range(100):
    new_vector = np.zeros((77, 768))
    for i in range(77):
        for j in range(768):
            new_vector[i, j] = random_value_in_range(min_values[i, j], max_values[i, j])
    new_latent_vectors_random_range.append(new_vector.reshape(1, 77, 768))

# Optionally, save these vectors as .npy files
for idx, vec in enumerate(new_latent_vectors_random_range):
    np.save(f'/path/to/save/vectors/random_range/new_vector_random_range_{idx}.npy', vec)
