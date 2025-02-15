import os
import cv2
import albumentations as A

def augment_image(image_path, output_folder):
    """
    Augments an image with various transformations and saves the augmented images.

    Parameters:
    image_path (str): Path to the input image.
    output_folder (str): Path to the folder where augmented images will be saved.

    Returns:
    None
    """
    try:
        # Read image preserving original color
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            return
        
        # Ensure output folder exists with full permissions
        os.makedirs(output_folder, exist_ok=True)
        
        # Create individual transformation variations
        transforms = [
            A.Compose([A.HorizontalFlip(p=1.0)]),
            A.Compose([A.VerticalFlip(p=1.0)]),
            A.Compose([A.Rotate(limit=180, p=1.0)]),
            A.Compose([A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)]),
        ]
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Apply different augmentations
        for i, transform in enumerate(transforms):
            augmented = transform(image=image)['image']
            
            output_path = os.path.join(output_folder, f"{filename}_aug_{i}.tif")
            try:
                cv2.imwrite(output_path, augmented)
                print(f"Saved augmented image: {output_path}")
            except PermissionError:
                print(f"Permission denied: {output_path}")
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images(input_folder, output_folder):
    """
    Processes all images in the input folder, applies augmentations, and saves them to the output folder.

    Parameters:
    input_folder (str): Path to the folder containing input images.
    output_folder (str): Path to the folder where augmented images will be saved.

    Returns:
    None
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        # Remove trailing space from input folder path
        input_folder = input_folder.rstrip()
        
        print(f"Input folder: {input_folder}")
        print(f"Contents of input folder: {os.listdir(input_folder)}")
        
        supported_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        
        for filename in os.listdir(input_folder):
            print(f"Processing file: {filename}")
            if filename.lower().endswith(supported_extensions):
                image_path = os.path.join(input_folder, filename)
                augment_image(image_path, output_folder)
            else:
                print(f"Unsupported file format: {filename}")
    
    except Exception as e:
        print(f"Error processing images in folder {input_folder}: {e}")

# Example usage
input_folder = "../../images/DATA SET/DIBaS/Acinetobacter.baumanii"
output_folder = "../../Augmented Images"
process_images(input_folder, output_folder)