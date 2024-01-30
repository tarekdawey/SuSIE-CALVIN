import numpy as np
import os

# This script prints the distances between each pair of vectors.

def load_vectors(directory):
    """
    Load vectors from .npy files in a given directory.

    Args:
    directory (str): The directory where .npy files are located.

    Returns:
    numpy.ndarray: An array of loaded vectors.
    """
    file_names = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
    vectors = [np.load(file_name) for file_name in file_names]
    return np.array(vectors)

def euclidean_distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two vectors.

    Args:
    vec1, vec2 (numpy.ndarray): Two vectors.

    Returns:
    float: The Euclidean distance between the vectors.
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def calculate_distances(vectors):
    """
    Calculate the pairwise Euclidean distances between a set of vectors.

    Args:
    vectors (numpy.ndarray): An array of vectors.

    Returns:
    numpy.ndarray: A matrix of pairwise distances.
    """
    num_vectors = vectors.shape[0]
    distance_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i+1, num_vectors):
            dist = euclidean_distance(vectors[i], vectors[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

# Load the vectors from the specified directory
directory = '/path/to/calvin/vectors'
vectors = load_vectors(directory)

# Calculate distances
distance_matrix = calculate_distances(vectors)

# Print the distances between each pair of vectors
for i in range(len(vectors)):
    for j in range(i+1, len(vectors)):
        print(f"Distance between vector {i+1} and vector {j+1}: {distance_matrix[i][j]}")
