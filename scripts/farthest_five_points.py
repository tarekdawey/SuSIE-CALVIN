import numpy as np
import os
from scipy.spatial.distance import cdist

def load_vectors(directory):
    file_names = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
    vectors = [np.load(file_name) for file_name in file_names]
    return np.array(vectors), file_names

def calculate_distances(new_vectors, original_vectors):
    distances = cdist(new_vectors.reshape(len(new_vectors), -1), original_vectors.reshape(len(original_vectors), -1), 'euclidean')
    min_distances = distances.min(axis=1)
    closest_original_indices = distances.argmin(axis=1)
    return min_distances, closest_original_indices

# Load original vectors
original_directory = '/path/to/calvin/vectors'
original_vectors, original_names = load_vectors(original_directory)

# Load new vectors
new_directory = '/path/to/random_vectors_min_max'
new_vectors, new_names = load_vectors(new_directory)

# Calculate distances
min_distances, closest_original_indices = calculate_distances(new_vectors, original_vectors)

# Find the 5 new vectors farthest away from any of the original vectors
farthest_indices = np.argsort(-min_distances)[:5]

# Print the names of the farthest new vectors and their corresponding closest original vector
for idx in farthest_indices:
    new_vector_name = os.path.basename(new_names[idx])
    closest_original_vector_name = os.path.basename(original_names[closest_original_indices[idx]])
    print(f"New Vector: {new_vector_name}, Closest Original Vector: {closest_original_vector_name}, Distance: {min_distances[idx]}")

