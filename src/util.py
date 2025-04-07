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

def move_and_rename_images(dir: str) -> None:
    """
    Move and rename all images from subdirectories within the `dir` folder to the main `dibas` directory.
    Each image is renamed as: subfoldername_filename.ext
    """
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                    old_path = os.path.join(subdir_path, file)

                    # New filename: subfoldername_filename.ext
                    base, ext = os.path.splitext(file)
                    new_name = f"{subdir}_{base}{ext}"
                    new_path = os.path.join(dir, new_name)

                    # Handle duplicates if they still occur
                    counter = 1
                    while os.path.exists(new_path):
                        new_name = f"{subdir}_{base}_{counter}{ext}"
                        new_path = os.path.join(dir, new_name)
                        counter += 1

                    shutil.move(old_path, new_path)
                    print(f"✅ Moved and Renamed: {file} -> {new_name}")

    print("🎉 All images have been moved and renamed in the main 'dibas' folder!")

# Example usage
# move_and_rename_images("data/raw/dibas")
