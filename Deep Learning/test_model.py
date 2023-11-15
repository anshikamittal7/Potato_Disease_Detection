import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_and_display_images(root_dir, num_images=3):
    # Get a list of all subdirectories in the root directory
    subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(subdirectories)

    # Create a single figure with multiple subplots
    fig, axs = plt.subplots(len(subdirectories), num_images, figsize=(12, 12))
    
    # Loop through each subdirectory
    for i, subdir in enumerate(subdirectories):
        subdir_path = os.path.join(root_dir, subdir)

        # Get a list of image files in the subdirectory
        image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.JPG'))]

        # Take at most 'num_images' images
        images_to_display = image_files[:num_images]

        # Display the images in the corresponding subplot
        for j, image_file in enumerate(images_to_display):
            image_path = os.path.join(subdir_path, image_file)
            img = mpimg.imread(image_path)

            # Display the image with the directory name as the title
            axs[i, j].imshow(img)
            axs[i, j].set_title(subdir)
            axs[i, j].axis('on')  # Turn off axis labels

            # Add line numbers
            # for line_num, line_text in enumerate(range(1, num_images + 1), start=1):
            #     axs[i, j].text(0, (line_num - 1) / num_images, str(line_text), transform=axs[i, j].transAxes,
            #                    color='white', fontsize=8, va='center', ha='left', fontweight='bold')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    plt.show()

# Example usage: Replace 'your_root_directory' with the actual path to your root directory
load_and_display_images('Data/training', num_images=3)
