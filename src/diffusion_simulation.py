import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
import argparse
import os
import sys
from numba import njit
import time

class DiffusionSimulation:
    def __init__(self, image_path, diffusion_rate=0.1, source_value=1.0, show_space_only=False):
        """
        Initialize the diffusion simulation.
        
        Args:
            image_path: Path to the blueprint image
            diffusion_rate: Rate of diffusion (between 0 and 1)
            source_value: Concentration value at source points
            show_space_only: If True, only show the space and entry points without running simulation
        """
        self.image_path = image_path
        self.diffusion_rate = diffusion_rate
        self.source_value = source_value
        self.show_space_only = show_space_only
        
        # Load and process image
        self.load_image()
        
        # Initialize concentration grid
        self.concentration = np.zeros_like(self.traversable_mask, dtype=float)
        
        # Manually define entry point (door on the left)
        self.set_entry_point()
        
        # Set concentration at entry point (door on the left)
        self.set_source_concentration()
    
    def load_image(self):
        """Load and process the blueprint image to create traversable and obstacle masks based on red boundary."""
        # Read the image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image at {self.image_path}")
        
        # Convert to RGB (from BGR)
        self.original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Extract red boundaries
        # Define the red color range (adjusting to better detect the red boundary)
        lower_red1 = np.array([150, 0, 0])    # First range of red
        upper_red1 = np.array([255, 100, 100])
        
        # Create mask for red boundaries
        red_mask = cv2.inRange(self.original_rgb, lower_red1, upper_red1)
        
        # Dilate to make sure the boundary is continuous
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        
        # Create a temporary image to work with
        h, w = red_mask.shape
        temp_image = np.zeros((h, w), dtype=np.uint8)
        temp_image[red_mask > 0] = 255
        
        # Create a copy of the image for contour finding
        contour_image = temp_image.copy()
        
        # Find contours
        contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for the area inside the red boundary
        traversable_mask = np.zeros_like(temp_image, dtype=bool)
        
        if contours:
            # Find the largest contour (should be the red boundary)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask of the inside of the contour
            mask = np.zeros_like(temp_image)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)  # -1 means fill
            
            # Set the traversable area to be inside the contour, excluding the boundary itself
            traversable_mask = (mask > 0) & (red_mask == 0)
        
        # Set the traversable mask
        self.traversable_mask = traversable_mask
        
        # Store the red boundary mask for entry point detection
        self.red_boundary_mask = red_mask > 0
        
        # Obstacle mask is the inverse of traversable_mask
        self.obstacle_mask = ~self.traversable_mask
        
        # Store dimensions
        self.height, self.width = self.traversable_mask.shape
        
        print(f"Image processed: {self.width}x{self.height}")
        print(f"Traversable area: {np.sum(self.traversable_mask)} pixels")
        
        # Save processed mask for debugging
        plt.figure(figsize=(12, 10))
        plt.imshow(self.traversable_mask, cmap='gray')
        plt.title('Traversable Area Mask (Inside Red Boundary)')
        plt.savefig("traversable_mask.png")
        plt.close()
    
    def set_entry_point(self):
        """Set the entry point using explicitly defined coordinates on the red boundary."""
        # User-specified coordinates from plot digitizer
        # Need to find the closest points on the red boundary to these coordinates
        user_points = [
            (384.98, 62.11),  # (y, x) format as used in our implementation
            (327.06, 63.16)   # (y, x) format
        ]
        
        # Convert to more convenient format with y-coordinate first (as our image indexing uses [y, x])
        # Note: The coordinates might need to be swapped depending on the coordinate system of the plot digitizer
        
        # Find all points on the red boundary
        boundary_points = []
        for y in range(self.height):
            for x in range(self.width):
                if self.red_boundary_mask[y, x]:
                    boundary_points.append((y, x))
        
        # Find the closest boundary points to the user-specified points
        closest_boundary_points = []
        for user_y, user_x in user_points:
            closest_dist = float('inf')
            closest_point = None
            
            for bound_y, bound_x in boundary_points:
                # Calculate Euclidean distance
                dist = (bound_y - user_y)**2 + (bound_x - user_x)**2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = (bound_y, bound_x)
            
            if closest_point:
                closest_boundary_points.append(closest_point)
            else:
                # Fallback if no boundary point is found (shouldn't happen)
                closest_boundary_points.append((int(user_y), int(user_x)))
        
        # If we found two boundary points, use them to define the entry area
        if len(closest_boundary_points) == 2:
            # Calculate the midpoint between the two boundary points
            y1, x1 = closest_boundary_points[0]
            y2, x2 = closest_boundary_points[1]
            
            # Set the entry point as the midpoint
            mid_y = (y1 + y2) // 2
            mid_x = (x1 + x2) // 2
            
            # Find a traversable point near this midpoint
            found_traversable = False
            search_radius = 10
            
            while not found_traversable and search_radius < 30:
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        ny, nx = mid_y + dy, mid_x + dx
                        if (0 <= ny < self.height and 0 <= nx < self.width and 
                            self.traversable_mask[ny, nx]):
                            # Found a traversable point
                            self.entry_point = (ny, nx)
                            found_traversable = True
                            break
                    if found_traversable:
                        break
                
                # Increase search radius if no traversable point was found
                search_radius += 5
            
            # If still no traversable point was found, use the midpoint anyway
            if not found_traversable:
                self.entry_point = (mid_y, mid_x)
                
            # Store the boundary points for visualization and concentration setting
            self.boundary_source_points = closest_boundary_points
        else:
            # Fallback if we don't have exactly two points
            # Just use the first boundary point we found or a default
            if boundary_points:
                self.entry_point = boundary_points[0]
                self.boundary_source_points = [boundary_points[0]]
            else:
                # Safe fallback - middle of left edge
                self.entry_point = (self.height // 2, 10)
                self.boundary_source_points = [self.entry_point]
        
        print(f"Entry point set at: {self.entry_point}")
        print(f"Boundary source points: {self.boundary_source_points}")
    
    def set_source_concentration(self):
        """Set the concentration along the specified boundary segment to a fixed value of 1.0.
        This represents a constant boundary condition where concentration is always 1.0 across the width."""
        # Use the boundary source points to define the source area
        if hasattr(self, 'boundary_source_points') and len(self.boundary_source_points) >= 2:
            # Get the two boundary points
            p1, p2 = self.boundary_source_points[0], self.boundary_source_points[1]
            y1, x1 = p1
            y2, x2 = p2
            
            # Calculate the number of points needed to interpolate between the two boundary points
            # This ensures a continuous line of source points along the boundary
            distance = int(np.sqrt((y2 - y1)**2 + (x2 - x1)**2))
            num_points = max(distance, 10)  # At least 10 points to ensure good coverage
            
            # Create a set of points along the line between p1 and p2
            for i in range(num_points + 1):
                # Linear interpolation between the two points
                t = i / num_points
                y = int((1 - t) * y1 + t * y2)
                x = int((1 - t) * x1 + t * x2)
                
                # Set concentration at this point and a small area around it
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.height and 0 <= nx < self.width and 
                            self.traversable_mask[ny, nx]):
                            # Set exact 1.0 at the boundary line and near it
                            self.concentration[ny, nx] = self.source_value
                
                # Also ensure we have some depth inward from the boundary
                # Find direction toward inside of the space (assume boundary is on the left)
                # This is a simplification - may need adjustment based on boundary orientation
                inward_x = x + 5  # Move 5 pixels inward (right)
                
                # Set concentration in a decreasing gradient inward
                for dist in range(1, 11):  # 10 pixels inward
                    nx = x + dist
                    if 0 <= nx < self.width:
                        # For a few pixels on each side of the point
                        for delta_y in range(-3, 4):
                            ny = y + delta_y
                            if (0 <= ny < self.height and 
                                self.traversable_mask[ny, nx]):
                                # Decreasing value as we move inward
                                value = self.source_value * (1 - 0.05 * dist)
                                self.concentration[ny, nx] = max(value, self.concentration[ny, nx])
        else:
            # Fallback: use the entry point and a fixed area around it
            y, x = self.entry_point
            
            # Define a rectangular area for the source
            entry_height = 20
            entry_width = 30
            
            # Set concentration in the entry area
            for dy in range(-entry_height//2, entry_height//2 + 1):
                for dx in range(-entry_width//2, entry_width//2 + 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width and 
                        self.traversable_mask[ny, nx]):
                        # Set entire entry area to exactly 1.0 (constant boundary)
                        self.concentration[ny, nx] = self.source_value
    
    def visualize_simulation_space(self):
        """
        Visualize the simulation space with the original image, traversable areas, and entry point.
        Creates an overlay image to help visualize the space.
        """
        # Create a copy of the original image for overlay
        overlay_img = self.original_rgb.copy()
        
        # Create a color mask for traversable areas (semi-transparent green)
        green_mask = np.zeros_like(overlay_img)
        green_mask[self.traversable_mask] = [0, 255, 0]  # Pure green for traversable
        
        # Blend the green mask with the original image
        alpha = 0.4  # Transparency factor
        overlay_img = cv2.addWeighted(overlay_img, 1-alpha, green_mask, alpha, 0)
        
        # Mark entry point based on boundary source points or fallback to entry point
        if hasattr(self, 'boundary_source_points') and len(self.boundary_source_points) >= 2:
            # Draw a line connecting the two boundary points to show the source area
            p1, p2 = self.boundary_source_points
            y1, x1 = p1
            y2, x2 = p2
            
            # Draw a thick red line along the boundary segment
            cv2.line(overlay_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Highlight the two boundary points
            cv2.circle(overlay_img, (x1, y1), 5, (255, 0, 0), -1)  # Filled circle
            cv2.circle(overlay_img, (x2, y2), 5, (255, 0, 0), -1)  # Filled circle
            
            # Mark the midpoint/entry point
            y, x = self.entry_point
            cv2.circle(overlay_img, (x, y), 7, (0, 0, 255), 2)  # Blue circle
            
            # Draw arrow to indicate entry direction (inward from boundary)
            # Calculate direction perpendicular to the boundary line
            dx = x2 - x1
            dy = y2 - y1
            # Perpendicular vector (pointing inward)
            perp_x = -dy
            perp_y = dx
            # Normalize and scale
            length = np.sqrt(perp_x**2 + perp_y**2)
            if length > 0:
                perp_x = int(20 * perp_x / length)
                perp_y = int(20 * perp_y / length)
                
                # Draw from mid-boundary point inward
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.arrowedLine(overlay_img, 
                              (mid_x, mid_y), 
                              (mid_x + perp_x, mid_y + perp_y), 
                              (255, 0, 0), 2, tipLength=0.3)
        else:
            # Fallback: mark the entry point with a rectangle
            y, x = self.entry_point
            entry_height = 20
            entry_width = 30
            cv2.rectangle(overlay_img, 
                         (x - entry_width//2, y - entry_height//2), 
                         (x + entry_width//2, y + entry_height//2), 
                         (255, 0, 0), 2)  # Red rectangle
            
            # Draw arrow
            arrow_length = 20
            cv2.arrowedLine(overlay_img, 
                           (x, y - arrow_length), 
                           (x, y), 
                           (255, 0, 0), 2, tipLength=0.3)
        
        # Create a figure with multiple plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Original image
        axes[0, 0].imshow(self.original_rgb)
        axes[0, 0].set_title('Original Blueprint')
        
        # Traversable mask
        axes[0, 1].imshow(self.traversable_mask, cmap='gray')
        axes[0, 1].set_title('Traversable Area Mask (Inside Red Boundary)')
        
        # Overlay view
        axes[1, 0].imshow(overlay_img)
        axes[1, 0].set_title('Overlay: Green = Traversable, Red = Source Location')
        
        # Initial concentration
        masked_concentration = np.ma.masked_where(~self.traversable_mask, self.concentration)
        im = axes[1, 1].imshow(masked_concentration, cmap='hot', vmin=0, vmax=self.source_value)
        axes[1, 1].set_title('Initial Concentration at Source Location')
        fig.colorbar(im, ax=axes[1, 1], label='Concentration')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("simulation_space_visualization.png", dpi=150)
        
        # Also display in a separate window if not in a headless environment
        plt.figure(figsize=(14, 12))
        plt.imshow(overlay_img)
        plt.title('Simulation Space: Green = Traversable, Red = Source Location')
        plt.savefig("simulation_space_overlay.png")
        
        print("\nSimulation space visualization saved as:")
        print("- simulation_space_visualization.png (4-panel view)")
        print("- simulation_space_overlay.png (overlay view)")
        
        return overlay_img
    
    def update_concentration(self):
        """Update the concentration grid using diffusion with vectorized operations."""
        # Create a copy of the current concentration grid
        new_concentration = self.concentration.copy()
        
        # Use the numba-accelerated diffusion function
        new_concentration = diffuse_numba(
            self.concentration,
            new_concentration,
            self.traversable_mask,
            self.diffusion_rate
        )
        
        # Update the concentration grid
        self.concentration = new_concentration
        
        # Apply fixed boundary condition at the door after updating the grid
        self.set_source_concentration()

    def run_simulation(self, steps=1000):
        """Run the diffusion simulation for a specified number of steps."""
        # First visualize the simulation space
        overlay_img = self.visualize_simulation_space()
        
        # If we're only showing the space, exit here
        if self.show_space_only:
            return
            
        # Ask for confirmation before starting the simulation
        confirm = input("\nDoes the simulation space look correct? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Simulation aborted. Please adjust parameters and try again.")
            return
        
        # Save initial state
        self.save_state("initial_state.png")
        
        # Performance tracking
        start_time = time.time()
        last_update_time = start_time
        
        # Run simulation steps
        step_batch = 1000  # Process this many steps between progress updates
        for step in range(0, steps, step_batch):
            # Process a batch of steps
            for i in range(min(step_batch, steps - step)):
                self.update_concentration()
            
            current_step = min(step + step_batch, steps)
            
            # Calculate and display performance metrics every batch
            current_time = time.time()
            elapsed = current_time - last_update_time
            steps_per_second = step_batch / max(elapsed, 0.001)
            
            # Update progress
            print(f"Step {current_step}/{steps} - {steps_per_second:.1f} steps/sec")
            
            # Save state at major milestones (only at 10000-step intervals)
            if current_step > 0 and current_step % 10000 == 0:
                self.save_state(f"state_step_{current_step}.png")
                
            last_update_time = current_time
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"\nSimulation completed in {total_time:.2f} seconds")
        print(f"Average performance: {steps/total_time:.1f} steps/second")
        
        # Save final state
        self.save_state("final_state.png")
    
    def save_state(self, filename):
        """Save the current state of the simulation."""
        plt.figure(figsize=(12, 10))
        
        # Create a masked version of concentration (only showing traversable areas)
        masked_concentration = np.ma.masked_where(~self.traversable_mask, self.concentration)
        
        # Plot the heatmap
        plt.imshow(masked_concentration, cmap='hot', vmin=0, vmax=self.source_value)
        plt.colorbar(label='Concentration')
        plt.title(f'Diffusion State - {filename}')
        plt.savefig(filename)
        plt.close()
    
    def generate_heatmap(self, output_path="diffusion_heatmap.png"):
        """Generate a heatmap visualization of the concentration distribution."""
        plt.figure(figsize=(12, 10))
        
        # Create a masked version of concentration (only showing traversable areas)
        masked_concentration = np.ma.masked_where(~self.traversable_mask, self.concentration)
        
        # Plot the heatmap
        plt.imshow(masked_concentration, cmap='hot', vmin=0, vmax=self.source_value)
        plt.colorbar(label='Concentration')
        plt.title('Diffusion Concentration Heatmap')
        plt.tight_layout()
        
        # Save the heatmap
        plt.savefig(output_path)
        print(f"Heatmap saved to {output_path}")
        plt.close()
    
    def animate_diffusion(self, output_path="diffusion_animation.mp4", frames=100, interval=50):
        """Create an animation of the diffusion process."""
        try:
            # Test if ffmpeg is available
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Check available writers
            writers = matplotlib.animation.writers.list()
            print(f"Available animation writers: {writers}")
            
            # Get the steps parameter from main or use default
            try:
                # Access steps from command line args
                parser = argparse.ArgumentParser()
                args, _ = parser.parse_known_args()
                total_steps = getattr(args, 'steps', 5000)
            except:
                # Fallback to default steps
                total_steps = 100000
            
            # Adjust number of frames based on the total steps to simulate
            # Ensure we have one frame for each major milestone (e.g., every 1000 steps)
            steps_per_frame = 100  # Each frame represents this many simulation steps
            frames = total_steps // steps_per_frame
            
            print(f"Animating {total_steps} steps across {frames} frames")
            
            # Reset concentration for animation
            self.concentration = np.zeros_like(self.traversable_mask, dtype=float)
            
            # Set entry point concentration
            self.set_source_concentration()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Pre-allocate the masked concentration array for better performance
            masked_concentration = np.ma.masked_where(~self.traversable_mask, self.concentration)
            
            # Initial plot
            im = ax.imshow(masked_concentration, cmap='hot', vmin=0, vmax=self.source_value)
            title = ax.set_title('Diffusion Simulation - Step 0')
            
            # More efficient update function
            def update(frame):
                # Update concentration multiple times per frame for faster visual diffusion
                # Use a local variable to avoid accessing self in a tight loop
                for _ in range(steps_per_frame):
                    self.update_concentration()
                
                current_step = (frame + 1) * steps_per_frame
                
                # Update only the data in the existing image rather than clearing and redrawing
                masked_concentration = np.ma.masked_where(~self.traversable_mask, self.concentration)
                im.set_array(masked_concentration)
                title.set_text(f'Diffusion Simulation - Step {current_step}')
                
                if frame % 10 == 0:
                    print(f"Animation frame {frame}/{frames} (step {current_step}/{total_steps})")
                    
                return [im, title]
            
            # Create animation with efficient blit=True
            ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
            
            # Try different writers in order of preference
            if 'ffmpeg' in writers:
                writer = 'ffmpeg'
            elif 'imagemagick' in writers:
                writer = 'imagemagick'
            elif 'pillow' in writers:
                writer = 'pillow'
            else:
                writer = None
                
            if writer:
                print(f"Using animation writer: {writer}")
                
                # Save animation with a higher dpi for better quality
                if writer == 'pillow':
                    # If using Pillow, save as GIF instead of MP4
                    gif_path = output_path.replace('.mp4', '.gif')
                    ani.save(gif_path, writer=writer, fps=10)
                    print(f"Animation saved to {gif_path}")
                else:
                    # Use a more optimal writer setup for ffmpeg
                    if writer == 'ffmpeg':
                        writer_obj = matplotlib.animation.FFMpegWriter(
                            fps=20,  # Higher FPS for smoother animation with many frames
                            metadata=dict(artist='DiffusionSimulation'),
                            bitrate=2000
                        )
                        ani.save(output_path, writer=writer_obj, dpi=100)
                    else:
                        ani.save(output_path, writer=writer, dpi=100)
                    print(f"Animation saved to {output_path}")
            else:
                print("No suitable animation writer found. Saving frame sequence instead.")
                # Save a sequence of frames instead, but more efficiently
                os.makedirs("animation_frames", exist_ok=True)
                
                # Adjust the frame count for the sequence
                sequence_frames = min(frames, 200)  # Limit to 200 frames to avoid too many files
                steps_per_sequence_frame = total_steps // sequence_frames
                
                for frame in range(sequence_frames):
                    # Update concentration for this frame
                    for _ in range(steps_per_sequence_frame):
                        self.update_concentration()
                    
                    current_step = (frame + 1) * steps_per_sequence_frame
                    
                    # Save current state more efficiently - only every 5th frame
                    if frame % 5 == 0 or frame == sequence_frames - 1:
                        frame_path = f"animation_frames/frame_{frame:04d}.png"
                        self.save_state(frame_path)
                        if frame % 20 == 0 or frame == sequence_frames - 1:
                            print(f"Saved frame {frame}/{sequence_frames} (step {current_step}/{total_steps})")
                
                print("Frame sequence saved to 'animation_frames/' directory")
                
            plt.close()
            
        except Exception as e:
            print(f"Animation error: {e}")
            print("Saving final state instead")
            self.save_state("final_diffusion_state.png")


@njit(nopython=True)
def diffuse_numba(concentration, new_concentration, traversable_mask, diffusion_rate):
    """Numba-accelerated version of the diffusion calculation."""
    height, width = concentration.shape
    
    # Manual implementation of the diffusion equation for each cell
    for y in range(1, height-1):
        for x in range(1, width-1):
            if traversable_mask[y, x]:
                # Only consider traversable neighbors
                neighbor_sum = 0.0
                neighbor_count = 0
                
                # Check each of the 4 neighbors
                if traversable_mask[y-1, x]:  # Up
                    neighbor_sum += concentration[y-1, x]
                    neighbor_count += 1
                if traversable_mask[y+1, x]:  # Down
                    neighbor_sum += concentration[y+1, x]
                    neighbor_count += 1
                if traversable_mask[y, x-1]:  # Left
                    neighbor_sum += concentration[y, x-1]
                    neighbor_count += 1
                if traversable_mask[y, x+1]:  # Right
                    neighbor_sum += concentration[y, x+1]
                    neighbor_count += 1
                
                # Apply diffusion equation if there are valid neighbors
                if neighbor_count > 0:
                    average = neighbor_sum / neighbor_count
                    new_concentration[y, x] = concentration[y, x] + diffusion_rate * (average - concentration[y, x])
    
    return new_concentration


    

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run diffusion simulation on a blueprint image')
    parser.add_argument('image_path', type=str, help='Path to the blueprint image')
    parser.add_argument('--diffusion_rate', type=float, default=0.1, help='Rate of diffusion (0-1)')
    parser.add_argument('--steps', type=int, default=5000, help='Number of simulation steps')
    parser.add_argument('--output', type=str, default='diffusion_heatmap.png', help='Output heatmap path')
    parser.add_argument('--animate', action='store_true', help='Create animation of diffusion')
    parser.add_argument('--animation_output', type=str, default='diffusion_animation.mp4', help='Output animation path')
    parser.add_argument('--show_space_only', action='store_true', help='Only show the simulation space without running the simulation')
    
    args = parser.parse_args()
    
    # Create and run simulation
    simulation = DiffusionSimulation(
        args.image_path,
        diffusion_rate=args.diffusion_rate,
        show_space_only=args.show_space_only
    )
    
    # Run the simulation
    simulation.run_simulation(steps=args.steps)
    
    # If we're only showing the space, exit here
    if args.show_space_only:
        return
    
    # Generate heatmap
    simulation.generate_heatmap(output_path=args.output)
    
    # Create animation if requested
    if args.animate:
        # Calculate appropriate frames based on steps
        total_steps = args.steps
        steps_per_frame = 100
        frames = max(50, total_steps // steps_per_frame)  # Ensure at least 50 frames
        
        print(f"Creating animation with {frames} frames for {total_steps} steps")
        simulation.animate_diffusion(
            output_path=args.animation_output,
            frames=frames
        )

if __name__ == "__main__":
    main() 