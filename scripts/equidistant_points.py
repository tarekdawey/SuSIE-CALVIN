import numpy as np

def create_equidistant_points(latent_vector_1, latent_vector_2, num_points, save_path):
    """
    Create and save equidistant points between two latent vectors.

    Args:
        latent_vector_1 (np.ndarray): First latent vector.
        latent_vector_2 (np.ndarray): Second latent vector.
        num_points (int): Number of equidistant points to create.
        save_path (str): Directory path to save the generated points.
    """
    # Ensure the latent vectors have the same shape
    if latent_vector_1.shape != latent_vector_2.shape:
        raise ValueError("The latent vectors do not have the same shape.")

    for i in range(1, num_points + 1):
        alpha = i / (num_points + 1)
        point = latent_vector_1 * (1 - alpha) + latent_vector_2 * alpha
        # Ensure the generated point has the same shape as the original vectors
        if point.shape == latent_vector_1.shape:
            np.save(f"{save_path}/equidistant_point_{i}.npy", point)
        else:
            raise ValueError(f"The generated point {i} does not match the shape of the original vectors.")

# Example usage
latent_vector_1_path = '/path/to/calvin/vector1.npy'
latent_vector_2_path = '/path/to/calvin/vector2.npy'
save_path = '/path/to/save/new_points'

latent_vector_1 = np.load(latent_vector_1_path)
latent_vector_2 = np.load(latent_vector_2_path)

create_equidistant_points(latent_vector_1, latent_vector_2, 5, save_path)

