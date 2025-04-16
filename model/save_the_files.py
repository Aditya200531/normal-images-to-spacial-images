import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from pynput import mouse
import time
from pynput import keyboard

from depth_anything_v2.dpt import DepthAnythingV2


def apply_perspective_shift_with_inpainting(image, depth_map, shift_scale=50, inpaint_radius=3,
                                            inpaint_method=cv2.INPAINT_NS):
    h, w = depth_map.shape

    # Normalize depth values and compute shift
    depth_min, depth_max = depth_map.min(), depth_map.max()
    if depth_min == depth_max:
        raise ValueError("Invalid depth map: min and max depth are equal.")
    shift_scale = 50 * np.std(depth_map)  # Higher variance results in larger shifts

    # Non-linear scaling (example with exponential scaling)
    depth_scaled = np.exp((depth_map - depth_min) / (depth_max - depth_min)) - 1
    depth_scaled /= depth_scaled.max()  # Normalize to range [0, 1]
    depth_shift = (depth_scaled - 0.5) * shift_scale

    # Generate shifted coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    x_shifted = x_coords + depth_shift
    x_shifted = np.clip(x_shifted, 0, w - 1).astype(np.float32)

    # Reflect padding for boundary preservation
    padded_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    x_shifted += 10
    y_coords_float = (y_coords + 10).astype(np.float32)

    # Apply remapping
    corrected_image = cv2.remap(padded_image, x_shifted, y_coords_float, interpolation=cv2.INTER_CUBIC)

    corrected_image = corrected_image[10:-10, 10:-10]  # Remove padding

    # Detect gaps (unmapped regions)
    if len(corrected_image.shape) == 3:
        mask = np.all(corrected_image == 0, axis=-1).astype(np.uint8)
    else:
        mask = (corrected_image == 0).astype(np.uint8)

    # Refine the mask (remove noise and small gaps)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply inpainting
    inpainted_image = cv2.inpaint(corrected_image, mask, inpaintRadius=inpaint_radius, flags=inpaint_method)

    return inpainted_image


def generate_high_accuracy_stereo_images(image, depth_map, disparity_scale=20, baseline_shift=1,
                                         gaussian_kernel_size=5):
    h, w = depth_map.shape

    # Normalize depth map to disparity
    depth_min, depth_max = depth_map.min(), depth_map.max()
    if depth_min == depth_max:
        raise ValueError("Invalid depth map: min and max depth are equal.")

    # Apply Gaussian smoothing to depth map
    smoothed_depth_map = cv2.GaussianBlur(depth_map, (gaussian_kernel_size, gaussian_kernel_size), 0)

    # Calculate disparity
    disparity = (smoothed_depth_map - depth_min) / (depth_max - depth_min) * disparity_scale

    # Generate meshgrid for pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate left and right coordinates
    left_x_coords = (x_coords - baseline_shift * disparity).astype(np.float32)
    right_x_coords = (x_coords + baseline_shift * disparity).astype(np.float32)
    y_coords_float = y_coords.astype(np.float32)

    # Pad the image to handle boundary conditions
    padded_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REFLECT)

    # Adjust coordinates for padding
    left_x_coords += 10
    right_x_coords += 10
    y_coords_float += 10

    # Remap the padded image
    left_image = cv2.remap(
        padded_image, left_x_coords, y_coords_float, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
    )
    right_image = cv2.remap(
        padded_image, right_x_coords, y_coords_float, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
    )

    # Crop the remapped images back to original size
    left_image = left_image[10:-10, 10:-10]
    right_image = right_image[10:-10, 10:-10]

    return left_image, right_image


class EyeTracker:
    def __init__(self):
        self.eye_x = 0
        self.eye_y = 0
        self.target_x = 0
        self.target_y = 0
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.mouse_listener.start()

    def on_mouse_move(self, x, y):
        screen_w, screen_h = 1920, 1080  # Adjust to your screen resolution
        self.target_x = (x - screen_w / 2) / (screen_w / 2)
        self.target_y = (y - screen_h / 2) / (screen_h / 2)

    def update(self):
        smoothing = 0.1
        self.eye_x += (self.target_x - self.eye_x) * smoothing
        self.eye_y += (self.target_y - self.eye_y) * smoothing


def apply_parallax(image, depth_map, eye_x, eye_y, parallax_strength=30):
    height, width = image.shape[:2]

    # Smooth the depth map
    depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)

    # Normalize depth map
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Create coordinate maps
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Calculate displacement
    x_shift = eye_x * depth_map_normalized * parallax_strength
    y_shift = eye_y * depth_map_normalized * parallax_strength

    # Apply the shift
    new_x = x_coords + x_shift
    new_y = y_coords + y_shift

    # Add border padding
    border = 100
    padded_image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT)

    # Adjust coordinates for padding
    new_x = np.clip(new_x + border, 0, width + 2 * border - 1)
    new_y = np.clip(new_y + border, 0, height + 2 * border - 1)

    # Remap with interpolation
    output = cv2.remap(padded_image,
                       new_x.astype(np.float32),
                       new_y.astype(np.float32),
                       cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REFLECT)

    # Remove padding
    output = output[border:-border, border:-border]

    # Create mask for gaps
    if len(image.shape) == 3:
        mask = np.all(output == 0, axis=2).astype(np.uint8)
    else:
        mask = (output == 0).astype(np.uint8)

    # Dilate the mask
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    # Fill gaps using inpainting
    if len(image.shape) == 3:
        output_fixed = output.copy()
        for i in range(3):
            channel = output[:, :, i]
            output_fixed[:, :, i] = cv2.inpaint(channel, mask, 5, cv2.INPAINT_TELEA)
        output = output_fixed
    else:
        output = cv2.inpaint(output, mask, 5, cv2.INPAINT_TELEA)

    return output


def create_side_by_side_view(left_image, right_image):
    """Create a side-by-side stereoscopic view."""
    # Ensure images are the same shape
    assert left_image.shape == right_image.shape, "Left and right images must have the same dimensions"

    # Create a side-by-side view
    h, w = left_image.shape[:2]
    side_by_side = np.zeros((h, w * 2, 3), dtype=np.uint8)

    # Convert grayscale to RGB if needed
    if len(left_image.shape) == 2:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2RGB)
    if len(right_image.shape) == 2:
        right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2RGB)

    # Ensure images are RGB
    if left_image.shape[2] != 3:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGRA2RGB)
    if right_image.shape[2] != 3:
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGRA2RGB)

    # Copy images to the side-by-side view
    side_by_side[:, :w] = left_image
    side_by_side[:, w:] = right_image

    return side_by_side


def save_output_images(original_image, depth_map, left_image, right_image, img_path):
    """Save all output images."""
    # Create directory based on the original filename
    file_name = os.path.basename(img_path)
    file_base = os.path.splitext(file_name)[0]
    output_dir = f"output_{file_base}"
    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    cv2.imwrite(f"{output_dir}/{file_base}_original.png", original_image)

    # Save depth map (ensure it's properly normalized)
    depth_colored = cv2.applyColorMap(
        cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
        cv2.COLORMAP_INFERNO
    )
    cv2.imwrite(f"{output_dir}/{file_base}_depth.png", depth_colored)

    # Save left and right views
    cv2.imwrite(f"{output_dir}/{file_base}_left.png", left_image)
    cv2.imwrite(f"{output_dir}/{file_base}_right.png", right_image)

    # Create and save side-by-side view
    side_by_side = create_side_by_side_view(left_image, right_image)
    cv2.imwrite(f"{output_dir}/{file_base}_side_by_side.png", side_by_side)

    print(f"Saved all output images to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 with Perspective Correction')
    parser.add_argument('--img-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for the model')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder')
    parser.add_argument('--save-outputs', action='store_true', help='Save output images')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load(f'../checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Read the single image
    raw_image = cv2.imread(args.img_path)
    if raw_image is None:
        print(f"Error reading image: {args.img_path}")
        exit(1)

    print(f"Image {args.img_path} read successfully. Shape: {raw_image.shape}")

    # Resize the image if it's too large
    max_size = 2000  # Set a maximum size for the longest dimension
    if max(raw_image.shape) > max_size:
        scale = max_size / max(raw_image.shape)
        new_size = (int(raw_image.shape[1] * scale), int(raw_image.shape[0] * scale))
        raw_image = cv2.resize(raw_image, new_size)
        print(f"Image resized to: {new_size}")

    depth = depth_anything.infer_image(raw_image, args.input_size)

    # Check if depth is valid
    if depth is None or np.all(depth == 0):
        print(f"Invalid depth map for image: {args.img_path}")
        exit(1)

    print(f"Depth map for {args.img_path} generated. Min: {depth.min()}, Max: {depth.max()}")

    # Normalize depth map
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_norm = depth_norm.astype(np.uint8)

    # Generate stereo pair
    left_image, right_image = generate_high_accuracy_stereo_images(raw_image, depth)

    # Save the outputs immediately
    save_output_images(raw_image, depth, left_image, right_image, args.img_path)
    print(f"Generated and saved stereo pair and depth map")

    # Initialize eye tracker
    eye_tracker = EyeTracker()

    # Interactive display loop
    print("Starting interactive display - Move mouse to control parallax, press 'Q' to quit")
    raw_image, depth_norm = raw_image, depth_norm

    while True:
        # Update eye position smoothly
        eye_tracker.update()

        # Apply parallax effect
        output = apply_parallax(raw_image, depth_norm,
                                eye_tracker.eye_x,
                                eye_tracker.eye_y)

        # Add a small delay to control frame rate
        time.sleep(0.016)  # Approximately 60 FPS

        # Resize for display
        max_display_size = 800
        h, w = output.shape[:2]
        if h > max_display_size or w > max_display_size:
            scale = max_display_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            output_display = cv2.resize(output, new_size)
        else:
            output_display = output

        # Display
        cv2.imshow('Parallax View (Move mouse to control, Q to quit)', output_display)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting display.")
            break

    cv2.destroyAllWindows()