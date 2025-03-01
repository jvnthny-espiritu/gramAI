import os
from PIL import Image
import torchvision.transforms as transforms

input_folder = "clinical-pb/PDSP Pics/FINAL NA"  
output_folder = "clinical-pb/augmented/aug PDSP"  
os.makedirs(output_folder, exist_ok=True)  

TARGET_WIDTH = 1221  
TARGET_HEIGHT = 814 


transform = transforms.Compose([
    transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH), interpolation=Image.BICUBIC),  
    transforms.RandomHorizontalFlip(), 
    transforms.RandomResizedCrop(size=(TARGET_HEIGHT, TARGET_WIDTH)),
])

num_augmented_per_image = 10  
for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)

    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image = Image.open(input_path).convert("RGB")  


        image_basename = os.path.splitext(file_name)[0]

        for i in range(1, num_augmented_per_image + 1):
            augmented_image = transform(image)  
            
            new_file_name = f"{image_basename} - IA({i}).jpg"  
            output_path = os.path.join(output_folder, new_file_name)
            
            augmented_image.save(output_path, format="JPEG", quality=95)  
            print(f"Saved {new_file_name} to {output_folder}")
