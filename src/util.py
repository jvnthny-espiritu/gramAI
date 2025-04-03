import os
import shutil

def move_images_to_main(dibas_dir: str) -> None:
    """
    Move all images from subdirectories within the `dibas` folder to the main `dibas` directory.
    """
    for subdir in os.listdir(dibas_dir):
        subdir_path = os.path.join(dibas_dir, subdir)
        
        if os.path.isdir(subdir_path):  # Check if it's a subdirectory
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):  # Include .tif images
                    old_path = os.path.join(subdir_path, file)
                    new_path = os.path.join(dibas_dir, file)
                    
                    # Handle duplicate filenames
                    if os.path.exists(new_path):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(new_path):
                            new_path = os.path.join(dibas_dir, f"{base}_{counter}{ext}")
                            counter += 1
                    
                    shutil.move(old_path, new_path)
                    print(f"✅ Moved: {file} -> {new_path}")

    print("🎉 All images have been moved to the main 'dibas' folder!")

# Example Usage
# move_images_to_main("data/raw/dibas")
