import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse
import os
import pickle
from matplotlib.widgets import Slider
import cv2


def load_preprocessed_space(layout_path):
    """Load preprocessed space data from a pickle file."""
    with open(layout_path, 'rb') as f:
        data = pickle.load(f)
    return data


class DiffusionSimulation:
    def __init__(self, space_data, diffusion_coeff=0.5, convection_x=0.0, convection_y=0.0):
        """Initialize the diffusion simulation with preprocessed space data."""
        # Simulation parameters
        self.diffusion_coeff = diffusion_coeff  # Diffusion coefficient
        self.convection_x = convection_x        # Convection in x direction
        self.convection_y = convection_y        # Convection in y direction
        
        # Load preprocessed data
        self.traversable_mask = space_data['traversable_mask']
        self.red_boundary_mask = space_data['red_boundary_mask']
        self.height = space_data['height']
        self.width = space_data['width']
        self.entry_point = space_data['entry_point']
        self.boundary_source_points = space_data['boundary_source_points']
        self.original_rgb = space_data['original_rgb']
        
        # Load obstacle masks
        if 'white_obstacle_mask' in space_data:
            self.white_obstacle_mask = space_data['white_obstacle_mask']
        if 'grey_obstacle_mask' in space_data:
            self.grey_obstacle_mask = space_data['grey_obstacle_mask']
        elif 'obstacle_mask' in space_data:
            self.obstacle_mask = space_data['obstacle_mask']
        
        # Initialize concentration grid
        self.concentration = np.zeros((self.height, self.width))
        
        # Set source points
        self.setup_source_points()
    
    def setup_source_points(self):
        """Set up source points as traversable pixels adjacent to the red boundary segment between endpoints."""
        self.source_points = []
        if self.boundary_source_points and len(self.boundary_source_points) == 2:
            # Get the two endpoints
            (y0, x0), (y1, x1) = self.boundary_source_points
            # Get all points along the segment using Bresenham's algorithm
            segment_pixels = self.bresenham_line(x0, y0, x1, y1)
            segment_boundary_pixels = set((y, x) for x, y in segment_pixels)
            # For every traversable pixel, check if it is adjacent to any boundary pixel in the segment
            for y in range(self.height):
                for x in range(self.width):
                    if self.traversable_mask[y, x] and not self.red_boundary_mask[y, x]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < self.height and 0 <= nx < self.width and
                                    (ny, nx) in segment_boundary_pixels):
                                    self.source_points.append((y, x))
                                    break
                            else:
                                continue
                            break
        elif self.boundary_source_points and len(self.boundary_source_points) > 2:
            # If boundary_source_points is a list of many points (e.g., full wall), treat all as the segment
            segment_boundary_pixels = set(self.boundary_source_points)
            for y in range(self.height):
                for x in range(self.width):
                    if self.traversable_mask[y, x] and not self.red_boundary_mask[y, x]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < self.height and 0 <= nx < self.width and
                                    (ny, nx) in segment_boundary_pixels):
                                    self.source_points.append((y, x))
                                    break
                            else:
                                continue
                            break
        # Fallback: if no source points found, use traversable pixels near entry_point
        if not self.source_points:
            y_entry, x_entry = self.entry_point
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    ny, nx = y_entry + dy, x_entry + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width and
                        self.traversable_mask[ny, nx] and not self.red_boundary_mask[ny, nx]):
                        self.source_points.append((ny, nx))
        # Make the list of source points unique
        self.source_points = list(set(self.source_points))
        # Set initial concentration for source points
        for y, x in self.source_points:
            self.concentration[y, x] = 1.0
        # Debug print
        print(f"[DEBUG] Number of source points: {len(self.source_points)}")
        print(f"[DEBUG] Source points: {self.source_points[:10]}{'...' if len(self.source_points) > 10 else ''}")
    
    def bresenham_line(self, x0, y0, x1, y1):
        """Implementation of Bresenham's line algorithm to get points along a line."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def step(self, dt=0.1):
        """Perform one step of the diffusion-convection simulation using the vectorized method."""
        self.step_vectorized(dt)
    
    def step_vectorized(self, dt=0.1):
        """Perform one step of the diffusion-convection simulation using vectorized operations."""
        # Create a valid cell mask (traversable and not boundary and not obstacle)
        valid_mask = self.traversable_mask & (~self.red_boundary_mask) & (~self.white_obstacle_mask if hasattr(self, 'white_obstacle_mask') else True)
        if hasattr(self, 'grey_obstacle_mask'):
            valid_mask = valid_mask & (~self.grey_obstacle_mask)
        # Remove border cells to avoid index errors
        valid_mask[0, :] = valid_mask[-1, :] = valid_mask[:, 0] = valid_mask[:, -1] = False
        
        # Create shifted versions of concentration array for Laplacian calculation
        c = self.concentration
        c_up = np.roll(c, -1, axis=0)    # Shift up
        c_down = np.roll(c, 1, axis=0)   # Shift down
        c_left = np.roll(c, -1, axis=1)  # Shift left
        c_right = np.roll(c, 1, axis=1)  # Shift right
        
        # Fix boundary conditions for shifted arrays
        c_up[-1, :] = c[-1, :]
        c_down[0, :] = c[0, :]
        c_left[:, -1] = c[:, -1]
        c_right[:, 0] = c[:, 0]
        
        # Create masks for valid neighbors
        up_valid = np.roll(valid_mask, 1, axis=0)    # If cell above is valid
        down_valid = np.roll(valid_mask, -1, axis=0) # If cell below is valid
        left_valid = np.roll(valid_mask, 1, axis=1)  # If cell to left is valid
        right_valid = np.roll(valid_mask, -1, axis=1)# If cell to right is valid
        
        # Fix boundary conditions for masks
        up_valid[0, :] = False
        down_valid[-1, :] = False
        left_valid[:, 0] = False
        right_valid[:, -1] = False
        
        # Create laplacian terms only for valid neighbors (blocking diffusion through obstacles)
        laplacian_x = np.zeros_like(c)
        laplacian_y = np.zeros_like(c)
        
        # Only add contribution from valid neighbors
        laplacian_x[valid_mask & left_valid] += c_left[valid_mask & left_valid] - c[valid_mask & left_valid]
        laplacian_x[valid_mask & right_valid] += c_right[valid_mask & right_valid] - c[valid_mask & right_valid]
        laplacian_y[valid_mask & up_valid] += c_up[valid_mask & up_valid] - c[valid_mask & up_valid]
        laplacian_y[valid_mask & down_valid] += c_down[valid_mask & down_valid] - c[valid_mask & down_valid]
        
        # Scale based on number of valid neighbors to maintain proportional diffusion
        valid_neighbor_count = (up_valid & valid_mask).astype(int) + (down_valid & valid_mask).astype(int) + \
                              (left_valid & valid_mask).astype(int) + (right_valid & valid_mask).astype(int)
        valid_neighbor_count = np.maximum(valid_neighbor_count, 1)  # Prevent division by zero
        diffusion = self.diffusion_coeff * (laplacian_x + laplacian_y)
        
        # Upwind convection scheme with obstacle blocking:
        # For X direction: if convection_x > 0 use c - c_left (backward), if < 0 use c_right - c (forward)
        # For Y direction: if convection_y > 0 use c - c_up (backward), if < 0 use c_down - c (forward)
        grad_x = np.zeros_like(c)
        grad_y = np.zeros_like(c)
        
        if self.convection_x > 0:
            grad_x[valid_mask & left_valid] = c[valid_mask & left_valid] - c_left[valid_mask & left_valid]
        elif self.convection_x < 0:
            grad_x[valid_mask & right_valid] = c_right[valid_mask & right_valid] - c[valid_mask & right_valid]
        
        if self.convection_y > 0:
            grad_y[valid_mask & up_valid] = c[valid_mask & up_valid] - c_up[valid_mask & up_valid]
        elif self.convection_y < 0:
            grad_y[valid_mask & down_valid] = c_down[valid_mask & down_valid] - c[valid_mask & down_valid]
        
        # No negative sign: convection moves in the direction of velocity
        convection = self.convection_x * grad_x + self.convection_y * grad_y
        
        # Update concentration for valid cells
        new_concentration = self.concentration.copy()
        update = dt * (diffusion + convection)
        new_concentration[valid_mask] += update[valid_mask]
        
        # Ensure concentration stays between 0 and 1
        new_concentration = np.clip(new_concentration, 0, 1)
        
        # Reset source points to maintain fixed concentration of 1.0
        for y, x in self.source_points:
            new_concentration[y, x] = 1.0
        
        # Update concentration
        self.concentration = new_concentration
    
    def get_concentration_grid(self):
        """Return the current concentration grid."""
        return self.concentration


def visualize_diffusion_simulation(space_data, steps=500, diffusion_coeff=0.5, 
                                  convection_x=0.0, convection_y=0.0, dt=0.1):
    """Visualize the diffusion simulation using the blueprint layout."""
    # Create simulation
    sim = DiffusionSimulation(
        space_data=space_data,
        diffusion_coeff=diffusion_coeff,
        convection_x=convection_x,
        convection_y=convection_y
    )
    
    # Set up the plot with styling
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 10))
    
    # Create grid for main plot and sliders
    gs = plt.GridSpec(16, 1)
    
    # Main plot takes up most of the space
    ax = fig.add_subplot(gs[0:12, 0])
    
    fig.patch.set_facecolor('#2A2A2A')
    ax.set_facecolor('#2A2A2A')
    
    # Set axes properties
    ax.set_xlim(0, sim.width)
    ax.set_ylim(sim.height, 0)  # Flip y-axis for image coordinates
    ax.set_aspect('equal')
    
    # Remove spines and ticks for minimalist look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title with custom styling
    ax.set_title('Diffusion-Convection Simulation', 
                 fontsize=18, fontweight='bold', color='white',
                 fontfamily='monospace', pad=20)
    
    # Display the original blueprint
    ax.imshow(sim.original_rgb, alpha=0.6)
    
    # Create mask for all obstacles to exclude from concentration display
    obstacle_mask = sim.red_boundary_mask.copy()
    if hasattr(sim, 'white_obstacle_mask'):
        obstacle_mask = obstacle_mask | sim.white_obstacle_mask
    if hasattr(sim, 'grey_obstacle_mask'):
        obstacle_mask = obstacle_mask | sim.grey_obstacle_mask
    elif hasattr(sim, 'obstacle_mask'):
        obstacle_mask = obstacle_mask | sim.obstacle_mask
    
    # Initialize the concentration plot with masked obstacles
    masked_concentration = np.ma.masked_array(
        sim.concentration, 
        mask=~sim.traversable_mask | obstacle_mask
    )
    concentration_plot = ax.imshow(
        masked_concentration, 
        cmap='viridis', 
        alpha=0.8,
        vmin=0, vmax=1
    )
    
    # Draw obstacles with distinct colors if available
    if hasattr(sim, 'white_obstacle_mask') and np.any(sim.white_obstacle_mask):
        white_y, white_x = np.where(sim.white_obstacle_mask)
        ax.scatter(white_x, white_y, color='magenta', s=5, alpha=0.6)
    
    if hasattr(sim, 'grey_obstacle_mask') and np.any(sim.grey_obstacle_mask):
        grey_y, grey_x = np.where(sim.grey_obstacle_mask)
        ax.scatter(grey_x, grey_y, color='blue', s=5, alpha=0.6)
    
    # Draw source points
    source_y = [y for y, x in sim.source_points]
    source_x = [x for y, x in sim.source_points]
    source_points_plot = ax.scatter(source_x, source_y, color='#FF5500', s=20, marker='o')
    
    # Add colorbar
    cbar = plt.colorbar(concentration_plot, ax=ax, pad=0.02)
    cbar.set_label('Concentration', color='white', fontfamily='monospace', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Status text
    status_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, 
        verticalalignment='top', fontsize=12, fontfamily='monospace',
        color='white', bbox=dict(facecolor='#2A2A2A', alpha=0.7, edgecolor='none')
    )
    
    # Add sliders for parameters
    # Diffusion coefficient slider
    ax_diffusion = fig.add_subplot(gs[12, 0])
    diffusion_slider = Slider(
        ax=ax_diffusion,
        label='Diffusion',
        valmin=0.0,
        valmax=1.0,
        valinit=diffusion_coeff,
        color='#4287f5'
    )
    
    # Convection X slider
    ax_convection_x = fig.add_subplot(gs[13, 0])
    convection_x_slider = Slider(
        ax=ax_convection_x,
        label='Convection X',
        valmin=-1.0,
        valmax=1.0,
        valinit=convection_x,
        color='#f54242'
    )
    
    # Convection Y slider
    ax_convection_y = fig.add_subplot(gs[14, 0])
    convection_y_slider = Slider(
        ax=ax_convection_y,
        label='Convection Y',
        valmin=-1.0,
        valmax=1.0,
        valinit=convection_y,
        color='#42f5a7'
    )
    
    # Time step slider
    ax_dt = fig.add_subplot(gs[15, 0])
    dt_slider = Slider(
        ax=ax_dt,
        label='Time Step',
        valmin=0.01,
        valmax=0.2,
        valinit=dt,
        valstep=0.01,
        color='#f5d742'
    )
    
    # Update function for sliders
    def update_params(val):
        sim.diffusion_coeff = diffusion_slider.val
        sim.convection_x = convection_x_slider.val
        sim.convection_y = convection_y_slider.val
    
    # Connect sliders to update function
    diffusion_slider.on_changed(update_params)
    convection_x_slider.on_changed(update_params)
    convection_y_slider.on_changed(update_params)
    
    def init():
        # Initial state
        # Create mask for all obstacles
        obstacle_mask = sim.red_boundary_mask.copy()
        if hasattr(sim, 'white_obstacle_mask'):
            obstacle_mask = obstacle_mask | sim.white_obstacle_mask
        if hasattr(sim, 'grey_obstacle_mask'):
            obstacle_mask = obstacle_mask | sim.grey_obstacle_mask
        elif hasattr(sim, 'obstacle_mask'):
            obstacle_mask = obstacle_mask | sim.obstacle_mask
            
        masked_concentration = np.ma.masked_array(
            sim.concentration, 
            mask=~sim.traversable_mask | obstacle_mask
        )
        concentration_plot.set_array(masked_concentration)
        status_text.set_text('')
        return [concentration_plot, status_text]
    
    def update(frame):
        # Run multiple simulation steps per frame
        steps_per_frame = 5
        current_dt = dt_slider.val
        
        for _ in range(steps_per_frame):
            sim.step(dt=current_dt)
        
        # Update concentration plot with obstacle masking
        obstacle_mask = sim.red_boundary_mask.copy()
        if hasattr(sim, 'white_obstacle_mask'):
            obstacle_mask = obstacle_mask | sim.white_obstacle_mask
        if hasattr(sim, 'grey_obstacle_mask'):
            obstacle_mask = obstacle_mask | sim.grey_obstacle_mask
        elif hasattr(sim, 'obstacle_mask'):
            obstacle_mask = obstacle_mask | sim.obstacle_mask
            
        masked_concentration = np.ma.masked_array(
            sim.concentration, 
            mask=~sim.traversable_mask | obstacle_mask
        )
        concentration_plot.set_array(masked_concentration)
        
        # Update status text
        status_text.set_text(
            f'STEP: {frame * steps_per_frame}\n'
            f'DIFFUSION: {sim.diffusion_coeff:.2f}\n'
            f'CONVECTION X: {sim.convection_x:.2f}\n'
            f'CONVECTION Y: {sim.convection_y:.2f}\n'
            f'DT: {current_dt:.2f}'
        )
        
        return [concentration_plot, status_text]
    
    anim = FuncAnimation(
        fig, update, frames=steps, init_func=init, 
        blit=True, interval=50, repeat=False
    )
    
    plt.tight_layout()
    plt.show()
    
    return sim


def generate_diffusion_heatmap(space_data, steps=1000, diffusion_coeff=0.5, 
                              convection_x=0.0, convection_y=0.0, dt=0.1):
    """Generate a heatmap of final concentration for the diffusion simulation."""
    
    # Create simulation
    sim = DiffusionSimulation(
        space_data=space_data,
        diffusion_coeff=diffusion_coeff,
        convection_x=convection_x,
        convection_y=convection_y
    )
    
    # Run the simulation
    for step in range(steps):
        sim.step(dt=dt)
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{steps}")
    
    # Set up the plot with styling
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('#2A2A2A')
    ax.set_facecolor('#2A2A2A')
    
    # Remove spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create full obstacle mask for visualization
    obstacle_mask = sim.red_boundary_mask.copy()
    if hasattr(sim, 'white_obstacle_mask'):
        obstacle_mask = obstacle_mask | sim.white_obstacle_mask
    if hasattr(sim, 'grey_obstacle_mask'):
        obstacle_mask = obstacle_mask | sim.grey_obstacle_mask
    elif hasattr(sim, 'obstacle_mask'):
        obstacle_mask = obstacle_mask | sim.obstacle_mask
    
    # Mask non-traversable areas in the concentration grid
    masked_concentration = np.ma.masked_array(
        sim.concentration, 
        mask=~sim.traversable_mask | obstacle_mask
    )
    
    # Create a composite visualization
    # First, show the original blueprint as background
    ax.imshow(sim.original_rgb, alpha=0.6)
    
    # Then overlay the heatmap
    heatmap = ax.imshow(masked_concentration, cmap='viridis', alpha=0.8, vmin=0, vmax=1)
    
    # Draw obstacles with distinct colors if available
    if hasattr(sim, 'white_obstacle_mask') and np.any(sim.white_obstacle_mask):
        white_y, white_x = np.where(sim.white_obstacle_mask)
        ax.scatter(white_x, white_y, color='magenta', s=5, alpha=0.6)
    
    if hasattr(sim, 'grey_obstacle_mask') and np.any(sim.grey_obstacle_mask):
        grey_y, grey_x = np.where(sim.grey_obstacle_mask)
        ax.scatter(grey_x, grey_y, color='blue', s=5, alpha=0.6)
    
    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax, pad=0.02)
    cbar.set_label('Concentration', color='white', fontfamily='monospace', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Draw source points
    source_y = [y for y, x in sim.source_points]
    source_x = [x for y, x in sim.source_points]
    ax.scatter(source_x, source_y, color='#FF5500', s=30, marker='o')
    
    # Add title and labels
    ax.set_title('Diffusion-Convection\nFinal Concentration Heatmap', 
                fontsize=18, fontweight='bold', color='white',
                fontfamily='monospace', pad=20)
    
    # Add simulation parameters as text
    param_text = (
        f'SIMULATION DETAILS:\n'
        f'DIFFUSION COEFF: {diffusion_coeff:.2f}\n'
        f'CONVECTION X: {convection_x:.2f}\n'
        f'CONVECTION Y: {convection_y:.2f}\n'
        f'TIME STEP: {dt:.2f}\n'
        f'STEPS: {steps}'
    )
    plt.figtext(0.02, 0.02, param_text, fontfamily='monospace', 
                color='white', fontsize=12)
    
    plt.tight_layout()
    # --- Save figure in 'figures' directory ---
    os.makedirs('figures', exist_ok=True)
    plt.savefig(os.path.join('figures', "diffusion_heatmap.png"), dpi=150)
    plt.show()
    
    return sim.concentration


def downsample_space_data(space_data, scale=0.25):
    """Downsample all relevant arrays in space_data by the given scale."""
    def resize_mask(mask):
        return cv2.resize(mask.astype(np.uint8), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)
    def resize_img(img):
        return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    downsampled = {}
    for key in ['traversable_mask', 'obstacle_mask', 'white_obstacle_mask', 'red_boundary_mask']:
        downsampled[key] = resize_mask(space_data[key])
    downsampled['original_rgb'] = resize_img(space_data['original_rgb'])
    downsampled['height'], downsampled['width'] = downsampled['traversable_mask'].shape
    # Scale entry and boundary points
    if space_data['entry_point'] is not None:
        downsampled['entry_point'] = tuple(int(round(x * scale)) for x in space_data['entry_point'])
    else:
        downsampled['entry_point'] = None
    downsampled['boundary_source_points'] = [
        tuple(int(round(x * scale)) for x in pt) for pt in space_data['boundary_source_points']
    ]
    return downsampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the diffusion simulation')
    parser.add_argument('--layout', type=str, required=True, help='Path to the preprocessed space layout file (pickle)')
    parser.add_argument('--diffusion', type=float, default=1.0, help='Diffusion coefficient')
    parser.add_argument('--convection_x', type=float, default=0.5, help='Convection in x direction')
    parser.add_argument('--convection_y', type=float, default=-0.1, help='Convection in y direction')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--steps', type=int, default=500, help='Number of simulation steps')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap instead of animation')
    parser.add_argument('--downsample', type=float, default=0.5, help='Downsample scale for development (e.g. 0.25)')
    args = parser.parse_args()
    space_data = load_preprocessed_space(args.layout)
    if args.downsample is not None:
        print(f"[INFO] Downsampling space data by scale {args.downsample}")
        space_data = downsample_space_data(space_data, scale=args.downsample)
    if args.heatmap:
        generate_diffusion_heatmap(
            space_data=space_data,
            steps=args.steps,
            diffusion_coeff=args.diffusion,
            convection_x=args.convection_x,
            convection_y=args.convection_y,
            dt=args.dt
        )
    else:
        visualize_diffusion_simulation(
            space_data=space_data,
            steps=args.steps,
            diffusion_coeff=args.diffusion,
            convection_x=args.convection_x,
            convection_y=args.convection_y,
            dt=args.dt
        )
