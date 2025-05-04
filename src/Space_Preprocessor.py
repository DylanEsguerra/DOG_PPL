# Space Preprocessor

#This script preprocesses the space by detecting boundaries and obstacles, and saving the processed space to a file.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import argparse
import cv2
import os
import sys
import matplotlib.transforms
import matplotlib.markers
import pickle
import matplotlib.widgets as mwidgets
from scipy.ndimage import label

class SpacePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.traversable_mask = None
        self.obstacle_mask = None
        self.white_obstacle_mask = None
        self.red_boundary_mask = None
        self.height = None
        self.width = None
        self.original_image = None
        self.original_rgb = None
        self.entry_point = None
        self.boundary_source_points = None

    def process_image(self):
        """Load and process the blueprint image to create traversable and obstacle masks based on red boundary, white, and grey obstacles."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image at {self.image_path}")
        self.original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        lower_red1 = np.array([150, 0, 0])
        upper_red1 = np.array([255, 100, 100])
        red_mask = cv2.inRange(self.original_rgb, lower_red1, upper_red1)
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        h, w = red_mask.shape
        temp_image = np.zeros((h, w), dtype=np.uint8)
        temp_image[red_mask > 0] = 255
        contour_image = temp_image.copy()
        contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        traversable_mask = np.zeros_like(temp_image, dtype=bool)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(temp_image)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            traversable_mask = (mask > 0) & (red_mask == 0)
        self.traversable_mask = traversable_mask
        self.red_boundary_mask = red_mask > 0
        self.height, self.width = self.traversable_mask.shape
        lower_white = np.array([220, 220, 220])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(self.original_rgb, lower_white, upper_white)
        white_obstacle_mask = (white_mask > 0) & self.traversable_mask
        self.white_obstacle_mask = white_obstacle_mask
        # --- Grey obstacle detection ---
        rgb = self.original_rgb
        min_rgb = np.min(rgb, axis=2)
        max_rgb = np.max(rgb, axis=2)
        mean_rgb = np.mean(rgb, axis=2)
        grey_mask = (
            (min_rgb >= 120) & (max_rgb <= 200) &
            ((max_rgb - min_rgb) < 20) & self.traversable_mask
        )
        self.grey_obstacle_mask = grey_mask
        # ---
        self.obstacle_mask = ~self.traversable_mask | self.white_obstacle_mask | self.grey_obstacle_mask
        # Filter white obstacles by area
        white_obstacle_mask_uint8 = white_obstacle_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(white_obstacle_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 100
        filtered_white_obstacle_mask = np.zeros_like(white_obstacle_mask, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                cv2.drawContours(filtered_white_obstacle_mask, [cnt], -1, 1, -1)
        self.white_obstacle_mask = filtered_white_obstacle_mask.astype(bool)
        # Filter grey obstacles by area
        grey_obstacle_mask_uint8 = grey_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(grey_obstacle_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 100
        filtered_grey_obstacle_mask = np.zeros_like(grey_mask, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                cv2.drawContours(filtered_grey_obstacle_mask, [cnt], -1, 1, -1)
        self.grey_obstacle_mask = filtered_grey_obstacle_mask.astype(bool)
        self.obstacle_mask = ~self.traversable_mask | self.white_obstacle_mask | self.grey_obstacle_mask
        print(f"Image processed: {self.width}x{self.height}")
        print(f"Traversable area: {np.sum(self.traversable_mask)} pixels")
        print(f"White obstacles detected: {np.sum(self.white_obstacle_mask)} pixels")
        print(f"Grey obstacles detected: {np.sum(self.grey_obstacle_mask)} pixels")

    def visualize_and_approve_obstacles(self):
        overlay = self.original_rgb.copy()
        # Function to update the overlay
        def update_overlay():
            overlay[:] = self.original_rgb
            overlay[self.white_obstacle_mask] = [255, 0, 255]  # magenta for white obstacles
            overlay[self.grey_obstacle_mask] = [100, 100, 255]  # blue for grey obstacles
            ax.imshow(overlay, origin='upper')
            fig.canvas.draw_idle()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        update_overlay()
        ax.set_aspect('equal')
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Detected Obstacles (magenta=white, blue=grey) over Blueprint\nClick magenta or blue to remove. Click "Done" when finished.', fontsize=14)
        plt.tight_layout()

        done = {'value': False}

        # Add a Done button
        button_ax = fig.add_axes([0.85, 0.01, 0.1, 0.05])
        done_button = mwidgets.Button(button_ax, 'Done', color='#cccccc', hovercolor='#aaaaaa')
        def on_done(event):
            done['value'] = True
            plt.close(fig)
        done_button.on_clicked(on_done)

        def on_click(event):
            if event.inaxes != ax:
                return
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < self.width and 0 <= y < self.height:
                # Remove white obstacle region
                if self.white_obstacle_mask[y, x]:
                    labeled, num = label(self.white_obstacle_mask)
                    region_label = labeled[y, x]
                    if region_label > 0:
                        self.white_obstacle_mask[labeled == region_label] = False
                        self.obstacle_mask = ~self.traversable_mask | self.white_obstacle_mask | self.grey_obstacle_mask
                        update_overlay()
                # Remove grey obstacle region
                elif self.grey_obstacle_mask[y, x]:
                    labeled, num = label(self.grey_obstacle_mask)
                    region_label = labeled[y, x]
                    if region_label > 0:
                        self.grey_obstacle_mask[labeled == region_label] = False
                        self.obstacle_mask = ~self.traversable_mask | self.white_obstacle_mask | self.grey_obstacle_mask
                        update_overlay()
        cid = fig.canvas.mpl_connect('button_press_event', on_click)

        # Wait for user to click Done
        while not done['value']:
            plt.pause(0.1)
        fig.canvas.mpl_disconnect(cid)
        # Final approval prompt
        resp = input("Do you want to save the processed space? (y/n): ")
        if resp.strip().lower() != 'y':
            print("Space not saved. Exiting.")
            sys.exit(1)

    def set_entry_point(self, user_points=None):
        if user_points is None:
            user_points = [
                (384.98, 63.0),
                (327.06, 63.0)
            ]
        boundary_points = []
        for y in range(self.height):
            for x in range(self.width):
                if self.red_boundary_mask[y, x]:
                    boundary_points.append((y, x))
        closest_boundary_points = []
        for user_y, user_x in user_points:
            closest_dist = float('inf')
            closest_point = None
            for bound_y, bound_x in boundary_points:
                dist = (bound_y - user_y)**2 + (bound_x - user_x)**2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = (bound_y, bound_x)
            if closest_point:
                closest_boundary_points.append(closest_point)
            else:
                closest_boundary_points.append((int(user_y), int(user_x)))
        if len(closest_boundary_points) == 2:
            y1, x1 = closest_boundary_points[0]
            y2, x2 = closest_boundary_points[1]
            mid_y = (y1 + y2) // 2
            mid_x = (x1 + x2) // 2
            found_traversable = False
            search_radius = 10
            while not found_traversable and search_radius < 30:
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        ny, nx = mid_y + dy, mid_x + dx
                        if (0 <= ny < self.height and 0 <= nx < self.width and 
                            self.traversable_mask[ny, nx]):
                            self.entry_point = (ny, nx)
                            found_traversable = True
                            break
                    if found_traversable:
                        break
                search_radius += 5
            if not found_traversable:
                self.entry_point = (mid_y, mid_x)
            self.boundary_source_points = closest_boundary_points
        else:
            if boundary_points:
                self.entry_point = boundary_points[0]
                self.boundary_source_points = [boundary_points[0]]
            else:
                self.entry_point = (self.height // 2, 10)
                self.boundary_source_points = [self.entry_point]
        print(f"Entry point set at: {self.entry_point}")
        print(f"Boundary source points: {self.boundary_source_points}")

    def save_to_file(self, out_path):
        data = {
            'traversable_mask': self.traversable_mask,
            'obstacle_mask': self.obstacle_mask,
            'white_obstacle_mask': self.white_obstacle_mask,
            'red_boundary_mask': self.red_boundary_mask,
            'height': self.height,
            'width': self.width,
            'entry_point': self.entry_point,
            'boundary_source_points': self.boundary_source_points,
            'original_rgb': self.original_rgb
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed space to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess a blueprint image and save the space layout.')
    parser.add_argument('image_path', type=str, help='Path to the blueprint image')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: layouts/processed_space.pkl)')
    args = parser.parse_args()
    if args.output is None:
        os.makedirs('layouts', exist_ok=True)
        args.output = os.path.join('layouts', 'processed_space.pkl')
    pre = SpacePreprocessor(args.image_path)
    pre.process_image()
    pre.visualize_and_approve_obstacles()
    pre.set_entry_point()
    pre.save_to_file(args.output)
