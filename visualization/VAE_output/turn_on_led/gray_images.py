import numpy as np
import matplotlib.pyplot as plt
import os

# Load the .npy array
image_array = np.load("/home/systemtec/calvin-sim/y_comp/y_array_6.npy")

# Ensure the shape is (1, 32, 32, 4)
if image_array.shape != (1, 32, 32, 4):
    raise ValueError("The shape of the loaded array is not (1, 32, 32, 4).")

# Create a directory to save the images if it doesn't exist
output_directory = "/home/systemtec/calvin-sim"
os.makedirs(output_directory, exist_ok=True)

# Extract each channel as a grayscale image and save them
for i in range(4):
    channel = image_array[0, :, :, i]  # Extract the channel
    # Construct the output file path
    output_file_path = os.path.join(output_directory, f"channel_{i + 1}.png")
    # Save the channel image as a PNG file
    plt.imsave(output_file_path, channel, cmap='gray')

# Print a message indicating where the images were saved
print(f"Grayscale images saved in: {output_directory}")
