# import torch
# import cv2
# import numpy as np
# from torchvision import transforms
# from PIL import Image
# import shutil

# # Load DPT-Small model from MiDaS
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
# model.eval()

# # Transformation pipeline
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5]),
# ])

# def generate_depth_map(image_path):
#     """Generate depth map using DPT-Small"""
#     img = Image.open(image_path).convert("RGB")
#     original_size = img.size  # (width, height)

#     # Resize image to 384x384 before processing
#     img_resized = img.resize((384, 384))
#     img_tensor = transform(img_resized).unsqueeze(0).to(device)  # Move to GPU if available

#     with torch.no_grad():
#         depth_map = model(img_tensor)
    
#     depth_map = depth_map.squeeze().cpu().numpy()

#     # Normalize depth map (0-255 range)
#     depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
#     depth_map = depth_map.astype(np.uint8)

#     # Resize depth map to original size
#     depth_img = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_CUBIC)
#     return depth_img

# def generate_right_view(left_image_path):
#     """Generate a synthetic right-eye view using depth estimation"""
#     left_img = cv2.imread(left_image_path)

#     if left_img is None:
#         print(f"❌ Error: Cannot open {left_image_path}. Check file path and format.")
#         return None

#     depth_map = generate_depth_map(left_image_path)

#     # Simulate right-eye view by shifting pixels based on depth
#     height, width, _ = left_img.shape
#     right_img = np.zeros_like(left_img)

#     for y in range(height):
#         for x in range(width):
#             shift = int((depth_map[y, x] / 255) * 15)  # Increase shift for better stereo effect
#             new_x = min(x + shift, width - 1)
#             right_img[y, new_x] = left_img[y, x]

#     return right_img

# def create_jps(left_image_path, output_jps_path):
#     """Create a JPS (stereoscopic JPEG) file"""
#     left_img = cv2.imread(left_image_path)

#     if left_img is None:
#         print(f"❌ Error: Cannot open {left_image_path}. Check file path and format.")
#         return
    
#     right_img = generate_right_view(left_image_path)

#     if right_img is None:
#         return

#     # Merge left and right images into a stereo pair
#     stereo_pair = np.hstack((left_img, right_img))

#     # Save as JPEG first
#     output_jpg = output_jps_path.replace(".jps", ".jpg")
#     cv2.imwrite(output_jpg, stereo_pair)

#     # Rename JPEG to .JPS
#     shutil.copy(output_jpg, output_jps_path)

#     print(f"✅ Stereo images saved as '{output_jpg}'.")
#     print(f"✅ JPS file saved as '{output_jps_path}' (renamed from JPEG).")

# # Convert JPEG to JPS using DPT-Small
# create_jps(r"C:\Users\Darshast5\Desktop\sp\spi_feb_2025\input1.jpeg", "output1.jps")


# with metadata

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import shutil
import piexif

# Load DPT-Small model from MiDaS for depth estimation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
model.eval()

# Transformation pipeline for input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def generate_depth_map(image_path):
    """Generate depth map using DPT-Small"""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (width, height)

    # Resize image to 384x384 before processing
    img_resized = img.resize((384, 384))
    img_tensor = transform(img_resized).unsqueeze(0).to(device)  # Move to GPU if available

    with torch.no_grad():
        depth_map = model(img_tensor)
    
    depth_map = depth_map.squeeze().cpu().numpy()

    # Normalize depth map (0-255 range)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map = depth_map.astype(np.uint8)

    # Resize depth map to original size
    depth_img = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_CUBIC)
    return depth_img

def generate_right_view(left_image_path):
    """Generate a synthetic right-eye view using depth estimation"""
    left_img = cv2.imread(left_image_path)

    if left_img is None:
        print(f"❌ Error: Cannot open {left_image_path}. Check file path and format.")
        return None

    depth_map = generate_depth_map(left_image_path)

    # Simulate right-eye view by shifting pixels based on depth
    height, width, _ = left_img.shape
    right_img = np.zeros_like(left_img)

    for y in range(height):
        for x in range(width):
            shift = int((depth_map[y, x] / 255) * 15)  # Parallax effect based on depth
            new_x = min(x + shift, width - 1)
            right_img[y, new_x] = left_img[y, x]

    return right_img

def add_exif_metadata(image_path):
    """Add EXIF metadata for VR headset compatibility"""
    exif_dict = {"0th": {piexif.ImageIFD.Make: "Meta", piexif.ImageIFD.Model: "Quest"}}
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)

def create_stereo_images(left_image_path, output_filename="output_3D_LR"):
    """Create a JPS (stereoscopic JPEG) and JPG file for VR viewing"""
    left_img = cv2.imread(left_image_path)

    if left_img is None:
        print(f"❌ Error: Cannot open {left_image_path}. Check file path and format.")
        return
    
    right_img = generate_right_view(left_image_path)

    if right_img is None:
        return

    # Merge left and right images into a stereo pair
    stereo_pair = np.hstack((left_img, right_img))

    # Define output filenames
    output_jpg = f"{output_filename}.jpg"
    output_jps = f"{output_filename}.jps"

    # Save as JPEG first
    cv2.imwrite(output_jpg, stereo_pair)
    
    # Add EXIF metadata
    add_exif_metadata(output_jpg)

    # Rename JPEG to .JPS
    shutil.copy(output_jpg, output_jps)

    print(f"✅ Stereo 3D image saved as '{output_jpg}'.")
    print(f"✅ JPS file saved as '{output_jps}'.")

# Convert JPEG to JPS using DPT-Small
create_stereo_images(r"C:\Users\Darshast5\Desktop\sp\spi_feb_2025\input.jpeg")
