import numpy as np
import os

def calculate_and_save_mean_latent_vector(source_directory, save_directory, save_file_name="mean_vector.npy"):
    """
    Calculate the mean latent vector from .npy files in a given directory and save the mean vector as a .npy file.

    Args:
    source_directory (str): The directory where .npy files are located.
    save_directory (str): The directory where the mean vector .npy file will be saved.
    save_file_name (str, optional): The name of the file to save the mean vector. Default is 'mean_vector.npy'.

    Returns:
    str: The path to the saved mean vector file.
    """
    # List all .npy files in the source directory
    file_names = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if f.endswith('.npy')]

    if not file_names:
        raise ValueError("No .npy files found in the source directory")

    print(f"Found {len(file_names)} .npy files in the source directory.")

    # Load the first file to initialize the sum_vector with the correct shape
    sum_vector = np.load(file_names[0])
    print(f"Loaded file: {file_names[0]}")

    # Accumulate the sum of vectors from the remaining files
    for file_name in file_names[1:]:
        vector = np.load(file_name)
        sum_vector += vector
        print(f"Loaded file: {file_name}")

    # Calculate the mean vector
    mean_vector = sum_vector / len(file_names)

    # Save the mean vector to the specified directory
    save_path = os.path.join(save_directory, save_file_name)
    np.save(save_path, mean_vector)

    print(f"Mean vector saved at {save_path}")

    return save_path

# Paths for demonstration
source_directory = '/path/to/calvin/vectors'
save_directory = '/path/to/save/the_mean_point'

# Uncomment and run this in your local environment where the directories are accessible.
saved_file_path = calculate_and_save_mean_latent_vector(source_directory, save_directory)
