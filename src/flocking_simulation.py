import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import argparse
import cv2
import os
import sys
import matplotlib.transforms
import matplotlib.markers
from matplotlib.widgets import Slider
import pickle
import matplotlib.colors as mcolors
from skimage.measure import block_reduce


def load_preprocessed_space(layout_path):
    """Load preprocessed space data from a pickle file."""
    with open(layout_path, 'rb') as f:
        data = pickle.load(f)
    return data


class FlockingDogParkSimulation:
    def __init__(self, space_data, cohesion_factor=0.5, 
                 separation_factor=0.5, alignment_factor=0.5, visual_range=20, death_rate=0.001):
        """Initialize the flocking dog park simulation with preprocessed space data."""
        self.cohesion_factor = cohesion_factor
        self.separation_factor = separation_factor
        self.alignment_factor = alignment_factor
        self.visual_range = visual_range
        self.max_speed = 2.0
        self.min_speed = 0.5
        self.max_force = 0.1
        self.arrival_rate = 0.05
        self.death_rate = death_rate
        # Load preprocessed data
        self.traversable_mask = space_data['traversable_mask']
        self.obstacle_mask = space_data['obstacle_mask']
        self.white_obstacle_mask = space_data['white_obstacle_mask']
        self.red_boundary_mask = space_data['red_boundary_mask']
        self.height = space_data['height']
        self.width = space_data['width']
        self.entry_point = space_data['entry_point']
        self.boundary_source_points = space_data['boundary_source_points']
        self.original_rgb = space_data['original_rgb']
        # Handle entry_point=None by picking a random boundary source point
        if self.entry_point is None and self.boundary_source_points:
            self.entry_y, self.entry_x = random.choice(self.boundary_source_points)
        elif self.entry_point is not None:
            self.entry_y, self.entry_x = self.entry_point
        else:
            raise ValueError("No entry point or boundary source points defined in layout.")
        self.agents = []
    
    def add_agent(self):
        """Add a new agent at the entry point or random boundary source point."""
        if self.entry_point is None and self.boundary_source_points:
            entry_y, entry_x = random.choice(self.boundary_source_points)
        else:
            entry_y, entry_x = self.entry_y, self.entry_x
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(self.min_speed, self.max_speed)
        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed
        self.agents.append([entry_x, entry_y, vx, vy])
    
    def apply_separation(self, agent_index, nearby_agents):
        """Calculate separation force to avoid other agents using vectorized operations."""
        if not nearby_agents:
            return 0, 0
        
        agent = self.agents[agent_index]
        agent_pos = np.array([agent[0], agent[1]])
        
        # Get positions of nearby agents
        nearby_positions = np.array([[self.agents[i][0], self.agents[i][1]] for i in nearby_agents])
        
        # Calculate direction vectors (from other agents to this agent)
        diff_vectors = agent_pos - nearby_positions
        
        # Calculate distances (with small epsilon to avoid division by zero)
        distances = np.sqrt(np.sum(diff_vectors**2, axis=1)) + 1e-8
        
        # Normalize direction vectors
        normalized_vectors = diff_vectors / distances[:, np.newaxis]
        
        # Weight by inverse square of distance
        weights = 1.0 / (distances**2)[:, np.newaxis]
        
        # Apply weights to normalized vectors
        weighted_vectors = normalized_vectors * weights
        
        # Sum all weighted vectors
        force = np.sum(weighted_vectors, axis=0)
        
        # Normalize and scale the resulting force
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > 0:
            force = (force / force_magnitude) * self.max_force * self.separation_factor
        
        return force[0], force[1]
    
    def apply_cohesion(self, agent_index, nearby_agents):
        """Calculate cohesion force to move toward other agents using vectorized operations."""
        if not nearby_agents:
            return 0, 0
        
        agent = self.agents[agent_index]
        agent_pos = np.array([agent[0], agent[1]])
        
        # Get positions of nearby agents
        nearby_positions = np.array([[self.agents[i][0], self.agents[i][1]] for i in nearby_agents])
        
        # Calculate center of mass
        com = np.mean(nearby_positions, axis=0)
        
        # Vector from current position to center of mass
        to_com = com - agent_pos
        
        # Distance to center of mass
        distance = np.linalg.norm(to_com) + 1e-8
        
        # Normalize and scale
        force = (to_com / distance) * self.max_force * self.cohesion_factor
        
        return force[0], force[1]
    
    def apply_alignment(self, agent_index, nearby_agents):
        """Calculate alignment force to match velocity with nearby agents using vectorized operations."""
        if not nearby_agents:
            return 0, 0
        
        agent = self.agents[agent_index]
        agent_velocity = np.array([agent[2], agent[3]])
        
        # Get velocities of nearby agents
        nearby_velocities = np.array([[self.agents[i][2], self.agents[i][3]] for i in nearby_agents])
        
        # Calculate average velocity
        avg_velocity = np.mean(nearby_velocities, axis=0)
        
        # Calculate alignment force (difference between average velocity and agent's velocity)
        force = (avg_velocity - agent_velocity) * self.alignment_factor
        
        # Limit the force magnitude
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > self.max_force:
            force = (force / force_magnitude) * self.max_force
        
        return force[0], force[1]
    
    def apply_boundary_avoidance(self, agent_index):
        """Force to avoid the non-traversable areas (boundaries and obstacles) using more efficient computation."""
        agent = self.agents[agent_index]
        x, y = int(agent[0]), int(agent[1])
        
        # Initialize force components
        force_x, force_y = 0, 0
        
        # Check surrounding area for non-traversable cells (including white obstacles)
        boundary_radius = 15  # Detection distance from boundaries/obstacles
        for dy in range(-boundary_radius, boundary_radius + 1):
            for dx in range(-boundary_radius, boundary_radius + 1):
                # Skip points outside our detection radius
                dist = np.sqrt(dx**2 + dy**2)
                if dist > boundary_radius or dist == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                # Check if point is within bounds and is an obstacle (boundary or white object)
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.obstacle_mask[ny, nx]):
                    # Apply repulsive force inversely proportional to distance
                    # Force points away from obstacle
                    repulsion = (boundary_radius - dist) / boundary_radius
                    force_x -= dx / dist * repulsion
                    force_y -= dy / dist * repulsion
        
        # Normalize and scale forces
        force_magnitude = np.sqrt(force_x**2 + force_y**2)
        if force_magnitude > 0:
            force_x = (force_x / force_magnitude) * self.max_force * 1.5  # Stronger than other forces
            force_y = (force_y / force_magnitude) * self.max_force * 1.5
        
        return force_x, force_y
    
    def find_nearby_agents(self, agent_index):
        """Find indexes of agents within visual range using vectorized operations."""
        agent = self.agents[agent_index]
        agent_pos = np.array([agent[0], agent[1]])
        
        # Convert all agent positions to numpy array
        all_positions = np.array([[a[0], a[1]] for a in self.agents])
        
        # Calculate distances using broadcasting
        distances = np.sqrt(np.sum((all_positions - agent_pos)**2, axis=1))
        
        # Create mask for agents within range (excluding self)
        mask = (distances <= self.visual_range) & (np.arange(len(self.agents)) != agent_index)
        
        # Return indices of nearby agents
        return np.where(mask)[0].tolist()
    
    def is_valid_position(self, x, y):
        """Check if a position is valid (within bounds and traversable, not in obstacle)."""
        # Convert to integer coordinates for mask lookup
        ix, iy = int(x), int(y)
        
        # Check if within bounds
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return False
        
        # Check if traversable and not an obstacle
        return self.traversable_mask[iy, ix] and not self.obstacle_mask[iy, ix]
    
    def limit_speed(self, vx, vy):
        """Limit speed to be between min_speed and max_speed using vectorized operations."""
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed > self.max_speed:
            # Scale down to max_speed
            return (vx / speed) * self.max_speed, (vy / speed) * self.max_speed
        elif speed < self.min_speed and speed > 0:
            # Scale up to min_speed
            return (vx / speed) * self.min_speed, (vy / speed) * self.min_speed
        else:
            return vx, vy
    
    def move_agents(self):
        """Move all agents using flocking behavior with optimized operations."""
        if not self.agents:
            return
        for i in range(len(self.agents)):
            nearby_agents = self.find_nearby_agents(i)
            separation_x, separation_y = self.apply_separation(i, nearby_agents)
            cohesion_x, cohesion_y = self.apply_cohesion(i, nearby_agents)
            alignment_x, alignment_y = self.apply_alignment(i, nearby_agents)
            boundary_x, boundary_y = self.apply_boundary_avoidance(i)
            self.agents[i][2] += separation_x + cohesion_x + alignment_x + boundary_x
            self.agents[i][3] += separation_y + cohesion_y + alignment_y + boundary_y
            self.agents[i][2], self.agents[i][3] = self.limit_speed(self.agents[i][2], self.agents[i][3])
            new_x = self.agents[i][0] + self.agents[i][2]
            new_y = self.agents[i][1] + self.agents[i][3]
            if self.is_valid_position(new_x, new_y):
                self.agents[i][0] = new_x
                self.agents[i][1] = new_y
            else:
                valid_move_found = False
                for attempt in range(8):
                    angle = 2 * np.pi * attempt / 8
                    test_x = self.agents[i][0] + np.cos(angle) * 2
                    test_y = self.agents[i][1] + np.sin(angle) * 2
                    if self.is_valid_position(test_x, test_y):
                        self.agents[i][0] = test_x
                        self.agents[i][1] = test_y
                        speed = np.sqrt(self.agents[i][2]**2 + self.agents[i][3]**2)
                        self.agents[i][2] = np.cos(angle) * speed
                        self.agents[i][3] = np.sin(angle) * speed
                        valid_move_found = True
                        break
                if not valid_move_found:
                    self.agents[i][2] *= -1
                    self.agents[i][3] *= -1
    
    def step(self):
        """Perform one step of the simulation."""
        if random.random() < self.arrival_rate and len(self.agents) < 200:
            self.add_agent()
        self.move_agents()
        # Remove agents with probability death_rate
        self.agents = [agent for agent in self.agents if random.random() > self.death_rate]
    
    def get_agent_positions(self):
        """Return the positions of all agents."""
        return np.array([[agent[0], agent[1]] for agent in self.agents])
    
    def get_agent_velocities(self):
        """Return numpy array of agent velocities"""
        velocities = np.array([agent[2:4] for agent in self.agents])
        return velocities
    
    def add_random_agents(self, n):
        """Add n agents at random valid locations with random velocity (not in obstacles or boundaries)."""
        valid_mask = self.traversable_mask & (~self.obstacle_mask) & (~self.red_boundary_mask)
        traversable_indices = np.argwhere(valid_mask)
        if len(traversable_indices) == 0:
            return
        chosen_indices = traversable_indices[np.random.choice(len(traversable_indices), size=n, replace=False)]
        for y, x in chosen_indices:
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(self.min_speed, self.max_speed)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed
            self.agents.append([x, y, vx, vy])


def visualize_flocking_simulation(space_data, steps=1000, arrival_rate=0.05, 
                                  cohesion_factor=0.5, separation_factor=0.5, 
                                  alignment_factor=0.5, perception_radius=20,
                                  max_force=0.1, death_rate=0.001):
    """Visualize the flocking simulation using the blueprint layout with triangles showing movement direction."""
    sim = FlockingDogParkSimulation(
        space_data=space_data,
        cohesion_factor=cohesion_factor,
        separation_factor=separation_factor,
        alignment_factor=alignment_factor,
        visual_range=perception_radius,
        death_rate=death_rate
    )
    sim.arrival_rate = arrival_rate
    sim.max_force = max_force
    sim.add_random_agents(100)
    
    # Set up the plot with new style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 10))
    
    # Create grid for main plot and sliders
    gs = plt.GridSpec(21, 1)  # 21 rows, 1 column (added one more for arrival rate slider)
    
    # Main plot takes up most of the space
    ax = fig.add_subplot(gs[0:16, 0])
    
    fig.patch.set_facecolor('#2A2A2A')
    ax.set_facecolor('#2A2A2A')
    
    # Set axes properties
    ax.set_xlim(0, sim.width)
    ax.set_ylim(0, sim.height)
    ax.set_aspect('equal')
    
    # Remove spines and ticks for minimalist look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title with custom styling
    ax.set_title('Traditional BOIDS Flocking Simulation', 
                 fontsize=18, fontweight='bold', color='white',
                 fontfamily='monospace', pad=20)
    
    # Create a mask for visualization - green for traversable, with alpha for visibility
    mask_img = np.zeros((sim.height, sim.width, 4))  # RGBA image
    # Green for traversable areas
    mask_img[sim.traversable_mask, 0] = 0.0    # R
    mask_img[sim.traversable_mask, 1] = 0.5    # G
    mask_img[sim.traversable_mask, 2] = 0.2    # B
    mask_img[sim.traversable_mask, 3] = 0.3    # Alpha (semi-transparent)
    
    # Red for boundaries
    mask_img[sim.red_boundary_mask, 0] = 0.8   # R
    mask_img[sim.red_boundary_mask, 1] = 0.1   # G
    mask_img[sim.red_boundary_mask, 2] = 0.1   # B
    mask_img[sim.red_boundary_mask, 3] = 0.5   # Alpha

    # Magenta for obstacles (including white obstacles and boundaries)
    mask_img[sim.obstacle_mask, 0] = 1.0   # R
    mask_img[sim.obstacle_mask, 1] = 0.0   # G
    mask_img[sim.obstacle_mask, 2] = 1.0   # B
    mask_img[sim.obstacle_mask, 3] = 0.5   # Alpha

    # Show the original blueprint with overlay
    ax.imshow(sim.original_rgb)
    ax.imshow(mask_img)
    
    # Draw the entry points
    if hasattr(sim, 'boundary_source_points') and len(sim.boundary_source_points) > 0:
        if len(sim.boundary_source_points) == 2:
            p1, p2 = sim.boundary_source_points
            y1, x1 = p1
            y2, x2 = p2
            # Draw a thick yellow line along the boundary segment
            ax.plot([x1, x2], [y1, y2], color='#FFFF00', linestyle='-', linewidth=3)
            # Highlight the entry point
            ax.scatter([sim.entry_x], [sim.entry_y], color='#FFFF00', s=100, marker='*')
        else:
            ys, xs = zip(*sim.boundary_source_points)
            ax.scatter(xs, ys, color='#FFFF00', s=10, marker='|')
    else:
        ax.scatter([sim.entry_x], [sim.entry_y], color='#FFFF00', s=100, marker='*')
    
    # Use a list to store artists that need to be cleaned up between frames
    artists = []
    
    def init():
        # Clear any existing artists
        for artist in artists:
            if artist in ax.get_children():
                artist.remove()
        artists.clear()
        
        # Reset the circle
        range_circle.set_center((0, 0))
        range_circle.set_visible(False)
        
        # Reset the text
        status_text.set_text('')
        
        return [range_circle, status_text]
    
    def update(frame):
        # Run multiple simulation steps per frame to speed up simulation
        steps_per_frame = 3
        for _ in range(steps_per_frame):
            sim.step()
        
        # Get agent positions and velocities
        positions = sim.get_agent_positions()
        velocities = sim.get_agent_velocities()
        
        # Clear previous triangles
        for artist in artists:
            if artist in ax.get_children():
                artist.remove()
        artists.clear()
        
        if len(positions) > 0:
            # Calculate angles
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            angles_deg = np.degrees(angles)
            
            # Create triangles with appropriate rotations
            for i, (x, y, angle) in enumerate(zip(positions[:, 0], positions[:, 1], angles_deg)):
                # Create a triangle pointing in direction of movement
                triangle = patches.RegularPolygon(
                    (x, y), 3, radius=5, 
                    orientation=np.radians(angle-90),  # -90 to make it point forward
                    color='white', alpha=0.7
                )
                ax.add_patch(triangle)
                artists.append(triangle)
            
            # Show visual range for first agent
            if len(sim.agents) > 0:
                range_circle.set_center((sim.agents[0][0], sim.agents[0][1]))
                range_circle.set_visible(True)
        else:
            range_circle.set_visible(False)
        
        # Update status text
        status_text.set_text(
            f'STEP: {frame * steps_per_frame}\nAGENTS: {len(sim.agents)}\n'
            f'COHESION: {sim.cohesion_factor:.1f}\n'
            f'SEPARATION: {sim.separation_factor:.1f}\n'
            f'ALIGNMENT: {sim.alignment_factor:.1f}\n'
            f'VISUAL RANGE: {sim.visual_range}\n'
            f'ARRIVAL: {sim.arrival_rate:.2f}'
        )
        
        # Return all artists that need to be redrawn
        return artists + [range_circle, status_text]
    
    # Visual range circle for reference
    range_circle = patches.Circle(
        (0, 0), radius=perception_radius, 
        fill=False, linestyle='--', edgecolor='#FFFFFF', alpha=0.4,
        visible=False
    )
    ax.add_patch(range_circle)
    
    # Status text with styling
    status_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, 
        verticalalignment='top', fontsize=12, fontfamily='monospace',
        color='white', bbox=dict(facecolor='#2A2A2A', alpha=0.7, edgecolor='none')
    )
    
    # Add sliders for parameters
    # Cohesion slider
    ax_cohesion = fig.add_subplot(gs[16, 0])
    cohesion_slider = Slider(
        ax=ax_cohesion,
        label='Cohesion',
        valmin=0.0,
        valmax=1.0,
        valinit=cohesion_factor,
        color='#4287f5'
    )
    
    # Separation slider
    ax_separation = fig.add_subplot(gs[17, 0])
    separation_slider = Slider(
        ax=ax_separation,
        label='Separation',
        valmin=0.0,
        valmax=1.0,
        valinit=separation_factor,
        color='#f54242'
    )
    
    # Alignment slider
    ax_alignment = fig.add_subplot(gs[18, 0])
    alignment_slider = Slider(
        ax=ax_alignment,
        label='Alignment',
        valmin=0.0,
        valmax=1.0,
        valinit=alignment_factor,
        color='#42f5a7'
    )
    
    # Perception radius slider
    ax_perception = fig.add_subplot(gs[19, 0])
    perception_slider = Slider(
        ax=ax_perception,
        label='Perception',
        valmin=5,
        valmax=50,
        valinit=perception_radius,
        valstep=1,
        color='#f5d742'
    )
    
    # Arrival rate slider
    ax_arrival_rate = fig.add_subplot(gs[20, 0])
    arrival_rate_slider = Slider(
        ax=ax_arrival_rate,
        label='Arrival Rate',
        valmin=0.0,
        valmax=1.0,
        valinit=arrival_rate,
        valstep=0.01,
        color='#f542f5'
    )
    
    # Update function for sliders
    def update_params(val):
        sim.cohesion_factor = cohesion_slider.val
        sim.separation_factor = separation_slider.val
        sim.alignment_factor = alignment_slider.val
        sim.visual_range = perception_slider.val
        sim.arrival_rate = arrival_rate_slider.val
        # Update the range circle
        range_circle.set_radius(perception_slider.val)
    
    # Connect sliders to update function
    cohesion_slider.on_changed(update_params)
    separation_slider.on_changed(update_params)
    alignment_slider.on_changed(update_params)
    perception_slider.on_changed(update_params)
    arrival_rate_slider.on_changed(update_params)
        
    anim = FuncAnimation(
        fig, update, frames=steps, init_func=init, 
        blit=True, interval=20, repeat=False, cache_frame_data=False
    )
    
    plt.tight_layout()
    plt.show()
    
    return sim


def generate_flocking_heatmap(space_data, steps=1000, arrival_rate=0.05, 
                             cohesion_factor=0.5, 
                             separation_factor=0.5, alignment_factor=0.5,
                             perception_radius=10, max_force=1.0, death_rate=0.001):
    """Generate a heatmap of dog density for the flocking simulation using blueprint layout."""
    sim = FlockingDogParkSimulation(
        space_data=space_data,
        cohesion_factor=cohesion_factor,
        separation_factor=separation_factor,
        alignment_factor=alignment_factor,
        visual_range=perception_radius,
        death_rate=death_rate
    )
    # --- Add: Load grey_obstacle_mask if present ---
    if 'grey_obstacle_mask' in space_data:
        sim.grey_obstacle_mask = space_data['grey_obstacle_mask']
    # --- End add ---
    sim.arrival_rate = arrival_rate
    sim.max_force = max_force
    sim.add_random_agents(100)
    
    # Initialize a density grid
    density_grid = np.zeros((sim.height, sim.width))
    
    # Run the simulation, tracking positions
    for step in range(steps):
        sim.step()
        
        # Record current agent positions in the density grid
        for agent in sim.agents:
            x, y = int(agent[0]), int(agent[1])
            if 0 <= x < sim.width and 0 <= y < sim.height:
                density_grid[y, x] += 1
                
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{steps} - {len(sim.agents)} agents")
    
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
    
    # Mask non-traversable areas in the density grid
    masked_density = np.ma.masked_array(density_grid, mask=~sim.traversable_mask)
    
    # Create a composite visualization
    # First, show the original blueprint as background, but more faded
    ax.imshow(sim.original_rgb, alpha=0.4)
    
    # Then overlay the heatmap with higher contrast
    norm = mcolors.PowerNorm(gamma=0.5)
    heatmap = ax.imshow(masked_density, cmap='plasma', alpha=0.85, interpolation='gaussian', norm=norm)
    
    # --- Add: Overlay obstacles as in diffusion_simulation.py ---
    if hasattr(sim, 'white_obstacle_mask') and np.any(sim.white_obstacle_mask):
        white_y, white_x = np.where(sim.white_obstacle_mask)
        ax.scatter(white_x, white_y, color='magenta', s=5, alpha=0.6)
    if hasattr(sim, 'grey_obstacle_mask') and np.any(sim.grey_obstacle_mask):
        grey_y, grey_x = np.where(sim.grey_obstacle_mask)
        ax.scatter(grey_x, grey_y, color='blue', s=5, alpha=0.6)
    # --- End add ---
    
    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax, pad=0.02)
    cbar.set_label('Agent Density', color='white', fontfamily='monospace', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Draw the entry points
    if hasattr(sim, 'boundary_source_points') and len(sim.boundary_source_points) > 0:
        if len(sim.boundary_source_points) == 2:
            p1, p2 = sim.boundary_source_points
            y1, x1 = p1
            y2, x2 = p2
            # Draw a thick yellow line along the boundary segment
            ax.plot([x1, x2], [y1, y2], color='#FFFF00', linestyle='-', linewidth=3)
            # Highlight the entry point
            ax.scatter([sim.entry_x], [sim.entry_y], color='#FFFF00', s=100, marker='*')
        else:
            ys, xs = zip(*sim.boundary_source_points)
            ax.scatter(xs, ys, color='#FFFF00', s=10, marker='|')
    else:
        ax.scatter([sim.entry_x], [sim.entry_y], color='#FFFF00', s=100, marker='*')
    
    # Add title and labels
    ax.set_title('Social Animals™\nDog Park Density Heatmap', 
                fontsize=18, fontweight='bold', color='white',
                fontfamily='monospace', pad=20)
    
    # Add simulation parameters as text
    param_text = (
        f'SIMULATION DETAILS:\n'
        f'COHESION: {cohesion_factor:.1f}\n'
        f'SEPARATION: {separation_factor:.1f}\n'
        f'ALIGNMENT: {alignment_factor:.1f}\n'
        f'STEPS: {steps}'
    )
    plt.figtext(0.02, 0.02, param_text, fontfamily='monospace', 
                color='white', fontsize=12)
    
    plt.tight_layout()
    # --- Save figure in 'figures' directory ---
    os.makedirs('figures', exist_ok=True)
    plt.savefig(os.path.join('figures', "flocking_heatmap.png"), dpi=150)
    plt.show()
    
    # --- Additional Plots ---
    # Downsample traversable and obstacle masks
    block_size = 5
    coarse_traversable = block_reduce(sim.traversable_mask, block_size=(block_size, block_size), func=np.any)
    coarse_white_obstacle = block_reduce(getattr(sim, 'white_obstacle_mask', np.zeros_like(sim.traversable_mask)), block_size=(block_size, block_size), func=np.any)
    coarse_grey_obstacle = block_reduce(getattr(sim, 'grey_obstacle_mask', np.zeros_like(sim.traversable_mask)), block_size=(block_size, block_size), func=np.any)

    # 1. Total Traveled Path Map (Binary, masked)
    visited = density_grid > 0
    coarse_visited = block_reduce(visited, block_size=(block_size, block_size), func=np.any)
    coarse_visited_masked = np.ma.masked_where(~coarse_traversable, coarse_visited)
    plt.figure(figsize=(20, 10))
    # Show blueprint as background, aligned
    plt.imshow(sim.original_rgb, alpha=0.4, extent=[0, sim.width, sim.height, 0], aspect='auto')
    # Overlay coarse visited mask, aligned
    plt.imshow(coarse_visited_masked, cmap='gray', interpolation='nearest', alpha=0.85,
               extent=[0, sim.width, sim.height, 0], aspect='auto')
    # Overlay obstacles, aligned
    if np.any(coarse_white_obstacle):
        plt.imshow(np.ma.masked_where(~coarse_white_obstacle, coarse_white_obstacle), cmap=mcolors.ListedColormap(['magenta']), alpha=0.5, interpolation='nearest', extent=[0, sim.width, sim.height, 0], aspect='auto')
    if np.any(coarse_grey_obstacle):
        plt.imshow(np.ma.masked_where(~coarse_grey_obstacle, coarse_grey_obstacle), cmap=mcolors.ListedColormap(['blue']), alpha=0.5, interpolation='nearest', extent=[0, sim.width, sim.height, 0], aspect='auto')
    # Overlay obstacle points for clarity, scaled
    if np.any(coarse_white_obstacle):
        y, x = np.where(coarse_white_obstacle)
        plt.scatter(x * block_size + block_size // 2, y * block_size + block_size // 2, color='magenta', s=10, alpha=0.7)
    if np.any(coarse_grey_obstacle):
        y, x = np.where(coarse_grey_obstacle)
        plt.scatter(x * block_size + block_size // 2, y * block_size + block_size // 2, color='blue', s=10, alpha=0.7)
    plt.title('Total Traveled Path (Visited Areas, Masked)')
    plt.axis('off')
    plt.tight_layout()
    # --- Save figure in 'figures' directory ---
    os.makedirs('figures', exist_ok=True)
    plt.savefig(os.path.join('figures', "flocking_total_traveled_path.png"), dpi=150)
    plt.show()
    
    # 2. Coarse-Grained, Normalized Heatmap (masked)
    coarse_density = block_reduce(density_grid, block_size=(block_size, block_size), func=np.sum)
    coarse_density_masked = np.ma.masked_where(~coarse_traversable, coarse_density)
    norm = mcolors.PowerNorm(gamma=0.5)
    plt.figure(figsize=(20, 10))
    # Show blueprint as background, aligned
    plt.imshow(sim.original_rgb, alpha=0.4, extent=[0, sim.width, sim.height, 0], aspect='auto')
    # Overlay coarse density heatmap, aligned
    plt.imshow(coarse_density_masked, cmap='plasma', norm=norm, interpolation='nearest', alpha=0.85,
               extent=[0, sim.width, sim.height, 0], aspect='auto')
    # Overlay obstacles, aligned
    if np.any(coarse_white_obstacle):
        plt.imshow(np.ma.masked_where(~coarse_white_obstacle, coarse_white_obstacle), cmap=mcolors.ListedColormap(['magenta']), alpha=0.5, interpolation='nearest', extent=[0, sim.width, sim.height, 0], aspect='auto')
    if np.any(coarse_grey_obstacle):
        plt.imshow(np.ma.masked_where(~coarse_grey_obstacle, coarse_grey_obstacle), cmap=mcolors.ListedColormap(['blue']), alpha=0.5, interpolation='nearest', extent=[0, sim.width, sim.height, 0], aspect='auto')
    # Overlay obstacle points for clarity, scaled
    if np.any(coarse_white_obstacle):
        y, x = np.where(coarse_white_obstacle)
        plt.scatter(x * block_size + block_size // 2, y * block_size + block_size // 2, color='magenta', s=10, alpha=0.7)
    if np.any(coarse_grey_obstacle):
        y, x = np.where(coarse_grey_obstacle)
        plt.scatter(x * block_size + block_size // 2, y * block_size + block_size // 2, color='blue', s=10, alpha=0.7)
    plt.title(f'Coarse-Grained Density Heatmap (block size {block_size}x{block_size}, Masked)')
    plt.colorbar(label='Agent Density')
    plt.axis('off')
    plt.tight_layout()
    # --- Save figure in 'figures' directory ---
    os.makedirs('figures', exist_ok=True)
    plt.savefig(os.path.join('figures', "flocking_coarse_grained_density_heatmap.png"), dpi=150)
    plt.show()
    # --- End Additional Plots ---
    
    return density_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the flocking dog park simulation')
    parser.add_argument('--layout', type=str, required=True, help='Path to the preprocessed space layout file (pickle)')
    parser.add_argument('--arrival_rate', type=float, default=0.1, help='Agent arrival rate (0-1)')
    parser.add_argument('--cohesion', type=float, default=0.5, help='Cohesion factor (0-1)')
    parser.add_argument('--separation', type=float, default=0.55, help='Separation factor (0-1)')
    parser.add_argument('--alignment', type=float, default=0.5, help='Alignment factor (0-1)')
    parser.add_argument('--perception', type=int, default=20, help='Perception radius')
    parser.add_argument('--max_force', type=float, default=1.0, help='Maximum steering force')
    parser.add_argument('--steps', type=int, default=2000, help='Number of simulation steps')
    parser.add_argument('--death_rate', type=float, default=0.001, help='Probability of agent death per step (0-1)')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap instead of animation')
    
    args = parser.parse_args()
    space_data = load_preprocessed_space(args.layout)
    if args.heatmap:
        generate_flocking_heatmap(
            space_data=space_data,
            arrival_rate=args.arrival_rate,
            steps=args.steps,
            cohesion_factor=args.cohesion,
            separation_factor=args.separation,
            alignment_factor=args.alignment,
            perception_radius=args.perception,
            max_force=args.max_force,
            death_rate=args.death_rate
        )
    else:
        visualize_flocking_simulation(
            space_data=space_data,
            steps=args.steps,
            arrival_rate=args.arrival_rate,
            cohesion_factor=args.cohesion,
            separation_factor=args.separation,
            alignment_factor=args.alignment,
            perception_radius=args.perception,
            max_force=args.max_force,
            death_rate=args.death_rate
        ) 