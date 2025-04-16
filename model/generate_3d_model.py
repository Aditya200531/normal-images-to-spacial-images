

import os
import sys
import cv2
import torch
import numpy as np
import pyvista as pv
import argparse
import glob
import matplotlib.pyplot as plt
from pyvistaqt import BackgroundPlotter
from PyQt5 import QtWidgets, QtCore
from depth_anything_v2.dpt import DepthAnythingV2


def mesh_norm_units(mesh, tex_img):
    """Calculate mesh normalization units"""
    scrn_cy, scrn_cx, _ = tex_img.shape
    mesh_cy, mesh_cx = mesh.shape

    mesh_unit_cx = scrn_cx / 2.0
    mesh_unit_cy = scrn_cy / 2.0

    mesh_unit = mesh_unit_cx if mesh_unit_cx > mesh_unit_cy else mesh_unit_cy

    mesh_unit_x = mesh_unit_cx / mesh_unit
    mesh_unit_y = mesh_unit_cy / mesh_unit

    mesh_step_x = mesh_unit_x / (mesh_cx / 2.0)
    mesh_step_y = mesh_unit_y / (mesh_cy / 2.0)

    return (mesh_unit_x, mesh_unit_y, mesh_step_x, mesh_step_y)


class ParallaxViewer:
    def __init__(self, plotter, n_layers):
        self.plotter = plotter
        self.n_layers = n_layers
        self.layer_spacing = 0.1
        self.parallax_strength = 0.5
        self.view_position = np.array([0.0, 0.0, 0.0])
        self.base_positions = {}
        self.movement_speed = 0.05
        self.setup_controls()
        self.store_base_positions()

    def store_base_positions(self):
        actors = self.plotter.renderer.actors
        for actor_key in actors.keys():
            if 'layer' in actor_key:
                actor = actors[actor_key]
                self.base_positions[actor_key] = np.array(actor.position)

    def setup_controls(self):
        self.plotter.add_key_event('Left', lambda: self.move_view(-1, 0))
        self.plotter.add_key_event('Right', lambda: self.move_view(1, 0))
        self.plotter.add_key_event('Up', lambda: self.move_view(0, 1))
        self.plotter.add_key_event('Down', lambda: self.move_view(0, -1))
        self.plotter.add_key_event('+', self.increase_parallax)
        self.plotter.add_key_event('-', self.decrease_parallax)
        self.plotter.add_key_event('f', self.increase_speed)
        self.plotter.add_key_event('s', self.decrease_speed)

    def move_view(self, dx, dy):
        self.view_position[0] += dx * self.movement_speed
        self.view_position[1] += dy * self.movement_speed
        self.update_layer_positions()

    def increase_parallax(self):
        self.parallax_strength *= 1.2
        self.update_layer_positions()
        print(f"Parallax strength: {self.parallax_strength:.2f}")

    def decrease_parallax(self):
        self.parallax_strength *= 0.8
        self.update_layer_positions()
        print(f"Parallax strength: {self.parallax_strength:.2f}")

    def increase_speed(self):
        self.movement_speed *= 1.2
        print(f"Movement speed: {self.movement_speed:.3f}")

    def decrease_speed(self):
        self.movement_speed *= 0.8
        print(f"Movement speed: {self.movement_speed:.3f}")

    def update_layer_positions(self):
        actors = self.plotter.renderer.actors
        for actor_key in actors.keys():
            if 'layer' in actor_key:
                layer_num = int(actor_key.split('_')[1])
                actor = actors[actor_key]
                base_pos = self.base_positions[actor_key]

                parallax_factor = (layer_num / self.n_layers)

                offset_x = self.view_position[0] * parallax_factor * self.parallax_strength
                offset_y = self.view_position[1] * parallax_factor * self.parallax_strength

                new_pos = (
                    base_pos[0] + offset_x,
                    base_pos[1] + offset_y,
                    base_pos[2]
                )

                actor.position = new_pos

        self.plotter.render()


def create_depth_layers(depth_norm, n_layers=5):
    """
    Segment depth map into discrete layers with overlap to prevent gaps
    """
    depth_min = depth_norm.min()
    depth_max = depth_norm.max()

    # Create overlapping boundaries
    overlap = 0.15  # 15% overlap between layers
    step = (depth_max - depth_min) / (n_layers - 1)
    layer_boundaries = []

    for i in range(n_layers):
        lower = max(depth_min, depth_min + i * step - overlap * step)
        upper = min(depth_max, depth_min + (i + 1) * step + overlap * step)
        layer_boundaries.append((lower, upper))


    layers = []
    for lower, upper in layer_boundaries:
        layer_mask = (depth_norm >= lower) & (depth_norm <= upper)
        if not layer_mask.any():  # Skip empty layers
            continue
        layers.append(layer_mask)

    return layers


def create_layer_mesh(x, y, depth_scaled, layer_mask, texture_img):
    """Create a separate mesh for each layer with soft edges"""
    # Create a distance-based falloff for the edges
    from scipy.ndimage import distance_transform_edt

    # Create soft edges for the mask
    dist_inside = distance_transform_edt(layer_mask)
    dist_outside = distance_transform_edt(~layer_mask)

    # Create a soft transition zone
    transition_width = 3
    alpha = 1.0 - np.clip((dist_outside - dist_inside) / transition_width, 0, 1)

    # Apply the soft mask to depth
    depth_layer = np.where(layer_mask, depth_scaled, np.nan)

    # Create mesh
    mesh_sg = pv.StructuredGrid(x, y, depth_layer)

    # Create masked texture with soft edges
    texture_masked = texture_img.copy()
    alpha_channel = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    texture_masked[..., 3] = alpha_channel
    non_nan_depth = depth_layer[~np.isnan(depth_layer)]
    if non_nan_depth.size == 0:
        raise ValueError(f"Empty layer detected at depth range. Skipping...")
    # Map texture coordinates
    mesh_sg.texture_map_to_plane(
        origin=(x.min(), y.min(), non_nan_depth.min()),
        point_u=(x.max(), y.min(), depth_layer[~np.isnan(depth_layer)].min()),
        point_v=(x.min(), y.max(), depth_layer[~np.isnan(depth_layer)].min()),
        inplace=True
    )

    texture = pv.numpy_to_texture(texture_masked)
    return mesh_sg, texture


def visualize_depth_mesh(image_path, encoder='vitl', input_size=518, n_layers=5):
    try:
        # Setup device
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        # Load model
        depth_anything = DepthAnythingV2(**model_configs[encoder])
        depth_anything.load_state_dict(
            torch.load(f'../checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()

        # Read and process image
        raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if raw_image.shape[2] == 3:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2BGRA)

        # Estimate depth
        with torch.no_grad():
            depth = depth_anything.infer_image(raw_image, input_size)

        # Create depth layers

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_norm = cv2.GaussianBlur(depth_norm, (5, 5), 0)

        layers = create_depth_layers(depth_norm, n_layers)

        # Create mesh coordinates
        depth_cy, depth_cx = depth_norm.shape
        max_depth = 0.6
        depth_scaled = depth_norm * max_depth

        mesh_unit_x, mesh_unit_y, mesh_step_x, mesh_step_y = mesh_norm_units(depth_norm, raw_image)
        x = np.arange(-mesh_unit_x, mesh_unit_x, mesh_step_x)
        y = np.arange(mesh_unit_y, -mesh_unit_y, -mesh_step_y)
        x, y = np.meshgrid(x, y)

        # Setup visualization
        if not QtWidgets.QApplication.instance():
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()

        plotter = BackgroundPlotter(window_size=(1200, 600))

        # Add layers
        for i, layer_mask in enumerate(layers):
            mesh_sg, texture = create_layer_mesh(x, y, depth_scaled, layer_mask, raw_image)
            z_position = i * 0.1
            mesh_sg.translate([0, 0, z_position])
            plotter.add_mesh(mesh_sg, texture=texture, opacity=1.0, name=f'layer_{i}')

        # Initialize viewer
        viewer = ParallaxViewer(plotter, n_layers)

        # Add help text
        help_text = """
        
        """
        plotter.add_text(help_text, position='upper_left')

        def close_window():
            plotter.close()

        plotter.add_key_event("q", close_window)
        plotter.show()
        app.exec_()

        return layers

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Interactive Parallax Depth Mesh Visualization')
    parser.add_argument('--img-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for the model')
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder')
    parser.add_argument('--n-layers', type=int, default=5, help='Number of depth layers')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if os.path.isfile(args.img_path):
        visualize_depth_mesh(
            args.img_path,
            encoder=args.encoder,
            input_size=args.input_size,
            n_layers=args.n_layers
        )



if __name__ == '__main__':
    main()