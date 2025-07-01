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
                 separation_factor=0.5, alignment_factor=0.5, visual_range=20, 
                 overcrowding_threshold=10):
        """Initialize the flocking dog park simulation with preprocessed space data."""
        self.cohesion_factor = cohesion_factor
        self.separation_factor = separation_factor
        self.alignment_factor = alignment_factor
        self.visual_range = visual_range
        self.max_speed = 2.0
        self.min_speed = 0.5
        self.max_force = 0.1
        self.arrival_rate = 0.05
        self.overcrowding_threshold = overcrowding_threshold
        
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
        
        # Overcrowding tracking
        self.agent_overcrowding_duration = {}  # agent_id -> consecutive steps overcrowded
        self.agent_ids = 0  # Counter for unique agent IDs
        self.overcrowding_history = []  # Store metrics over time
        self.step_count = 0
    
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
        
        # Agent format: [x, y, vx, vy, agent_id]
        agent_id = self.agent_ids
        self.agent_ids += 1
        self.agents.append([entry_x, entry_y, vx, vy, agent_id])
        
        # Initialize overcrowding tracking for this agent
        self.agent_overcrowding_duration[agent_id] = 0
    
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
        
        # Update overcrowding tracking
        self.update_overcrowding_duration()
        
        # Store overcrowding metrics for this step
        metrics = self.calculate_overcrowding_metrics()
        metrics['step'] = self.step_count
        self.overcrowding_history.append(metrics)
        self.step_count += 1
    
    def get_agent_positions(self):
        """Return the positions of all agents."""
        return np.array([[agent[0], agent[1]] for agent in self.agents])
    
    def get_agent_velocities(self):
        """Return numpy array of agent velocities"""
        velocities = np.array([agent[2:4] for agent in self.agents])
        return velocities
    
    def get_overcrowded_agents(self):
        """Return positions of agents that are currently overcrowded."""
        overcrowded_positions = []
        for i in range(len(self.agents)):
            nearby = self.find_nearby_agents(i)
            if len(nearby) > self.overcrowding_threshold:
                overcrowded_positions.append([self.agents[i][0], self.agents[i][1]])
        return np.array(overcrowded_positions)
    
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
            # Agent format: [x, y, vx, vy, agent_id]
            agent_id = self.agent_ids
            self.agent_ids += 1
            self.agents.append([x, y, vx, vy, agent_id])
            # Initialize overcrowding tracking for this agent
            self.agent_overcrowding_duration[agent_id] = 0

    def calculate_overcrowding_metrics(self):
        """Calculate various overcrowding metrics for current state."""
        if not self.agents:
            return {
                'overcrowding_percentage': 0,
                'average_neighbors': 0,
                'max_neighbors': 0,
                'overcrowded_count': 0,
                'total_agents': 0,
                'agents_with_sustained_overcrowding': 0,
                'average_overcrowding_duration': 0,
                'max_overcrowding_duration': 0
            }
        
        neighbor_counts = []
        overcrowded_agents = 0
        sustained_overcrowded = 0
        
        for i in range(len(self.agents)):
            nearby = self.find_nearby_agents(i)
            neighbor_count = len(nearby)
            neighbor_counts.append(neighbor_count)
            
            agent_id = self.agents[i][4]  # Get agent ID
            
            if neighbor_count > self.overcrowding_threshold:
                overcrowded_agents += 1
                # Check if this agent has been overcrowded for multiple steps
                if self.agent_overcrowding_duration[agent_id] > 5:  # 5+ consecutive steps
                    sustained_overcrowded += 1
        
        # Calculate duration statistics
        durations = list(self.agent_overcrowding_duration.values())
        avg_duration = np.mean(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        return {
            'overcrowding_percentage': (overcrowded_agents / len(self.agents)) * 100,
            'average_neighbors': np.mean(neighbor_counts),
            'max_neighbors': max(neighbor_counts) if neighbor_counts else 0,
            'overcrowded_count': overcrowded_agents,
            'total_agents': len(self.agents),
            'agents_with_sustained_overcrowding': sustained_overcrowded,
            'average_overcrowding_duration': avg_duration,
            'max_overcrowding_duration': max_duration
        }
    
    def update_overcrowding_duration(self):
        """Update overcrowding duration for each agent."""
        for i in range(len(self.agents)):
            agent_id = self.agents[i][4]
            nearby = self.find_nearby_agents(i)
            neighbor_count = len(nearby)
            
            if neighbor_count > self.overcrowding_threshold:
                # Agent is overcrowded, increment duration
                self.agent_overcrowding_duration[agent_id] += 1
            else:
                # Agent is not overcrowded, reset duration
                self.agent_overcrowding_duration[agent_id] = 0
    
    def get_overcrowding_statistics(self):
        """Get comprehensive overcrowding statistics over time."""
        if not self.overcrowding_history:
            return {}
        
        history = self.overcrowding_history
        
        # Calculate statistics over the entire simulation
        avg_overcrowding_pct = np.mean([h['overcrowding_percentage'] for h in history])
        max_overcrowding_pct = max([h['overcrowding_percentage'] for h in history])
        avg_neighbors = np.mean([h['average_neighbors'] for h in history])
        max_neighbors_ever = max([h['max_neighbors'] for h in history])
        
        return {
            'simulation_steps': len(history),
            'average_overcrowding_percentage': avg_overcrowding_pct,
            'peak_overcrowding_percentage': max_overcrowding_pct,
            'average_neighbors_over_time': avg_neighbors,
            'maximum_neighbors_ever': max_neighbors_ever,
            'final_metrics': history[-1] if history else {}
        }
    
    def is_system_overcrowded(self, threshold_percentage=20):
        """Check if the system is considered overcrowded based on current metrics."""
        metrics = self.calculate_overcrowding_metrics()
        return metrics['overcrowding_percentage'] > threshold_percentage


def visualize_flocking_simulation(space_data, steps=1000, arrival_rate=0.05, 
                                  cohesion_factor=0.5, separation_factor=0.5, 
                                  alignment_factor=0.5, perception_radius=20,
                                  max_force=0.1, overcrowding_threshold=10):
    """Visualize the flocking simulation using the blueprint layout with triangles showing movement direction."""
    sim = FlockingDogParkSimulation(
        space_data=space_data,
        cohesion_factor=cohesion_factor,
        separation_factor=separation_factor,
        alignment_factor=alignment_factor,
        visual_range=perception_radius,
        overcrowding_threshold=overcrowding_threshold
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
            
            # Get overcrowded agent positions
            overcrowded_positions = sim.get_overcrowded_agents()
            overcrowded_set = set()
            if len(overcrowded_positions) > 0:
                for opos in overcrowded_positions:
                    # Find which agents are overcrowded by matching positions
                    for i, pos in enumerate(positions):
                        if np.allclose(pos, opos, atol=0.1):
                            overcrowded_set.add(i)
            
            # Create triangles with appropriate rotations and colors
            for i, (x, y, angle) in enumerate(zip(positions[:, 0], positions[:, 1], angles_deg)):
                # Color overcrowded agents red, others white
                color = 'red' if i in overcrowded_set else 'white'
                alpha = 0.9 if i in overcrowded_set else 0.7
                
                # Create a triangle pointing in direction of movement
                triangle = patches.RegularPolygon(
                    (x, y), 3, radius=5, 
                    orientation=np.radians(angle-90),  # -90 to make it point forward
                    color=color, alpha=alpha
                )
                ax.add_patch(triangle)
                artists.append(triangle)
            
            # Show visual range for first agent
            if len(sim.agents) > 0:
                range_circle.set_center((sim.agents[0][0], sim.agents[0][1]))
                range_circle.set_visible(True)
        else:
            range_circle.set_visible(False)
        
        # Get overcrowding metrics
        overcrowding_metrics = sim.calculate_overcrowding_metrics()
        
        # Update status text with overcrowding information
        status_text.set_text(
            f'STEP: {frame * steps_per_frame}\nAGENTS: {len(sim.agents)}\n'
            f'COHESION: {sim.cohesion_factor:.1f}\n'
            f'SEPARATION: {sim.separation_factor:.1f}\n'
            f'ALIGNMENT: {sim.alignment_factor:.1f}\n'
            f'VISUAL RANGE: {sim.visual_range}\n'
            f'ARRIVAL: {sim.arrival_rate:.2f}\n'
            f'OVERCROWDED: {overcrowding_metrics["overcrowded_count"]}\n'
            f'OVERCROWD %: {overcrowding_metrics["overcrowding_percentage"]:.1f}%\n'
            f'AVG NEIGHBORS: {overcrowding_metrics["average_neighbors"]:.1f}\n'
            f'MAX NEIGHBORS: {overcrowding_metrics["max_neighbors"]}'
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
                             perception_radius=10, max_force=1.0,
                             overcrowding_threshold=10):
    """Generate a heatmap of dog density for the flocking simulation using blueprint layout."""
    sim = FlockingDogParkSimulation(
        space_data=space_data,
        cohesion_factor=cohesion_factor,
        separation_factor=separation_factor,
        alignment_factor=alignment_factor,
        visual_range=perception_radius,
        overcrowding_threshold=overcrowding_threshold
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
    
    # Print overcrowding statistics
    final_stats = sim.get_overcrowding_statistics()
    if final_stats:
        print("\n" + "=" * 60)
        print("📊 OVERCROWDING ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Simulation Steps: {final_stats['simulation_steps']}")
        print(f"Average Overcrowding: {final_stats['average_overcrowding_percentage']:.1f}%")
        print(f"Peak Overcrowding: {final_stats['peak_overcrowding_percentage']:.1f}%")
        print(f"Average Neighbors: {final_stats['average_neighbors_over_time']:.1f}")
        print(f"Maximum Neighbors: {final_stats['maximum_neighbors_ever']}")
        
        if final_stats['final_metrics']:
            final = final_stats['final_metrics']
            print(f"\nFinal State:")
            print(f"  Total Agents: {final['total_agents']}")
            print(f"  Overcrowded Agents: {final['overcrowded_count']}")
            print(f"  Sustained Overcrowding: {final['agents_with_sustained_overcrowding']}")
    
    return density_grid


def analyze_park_capacity(space_data, min_agents=20, max_agents=300, step_size=20, 
                         simulation_steps=500, cohesion_factor=0.5, 
                         separation_factor=0.5, alignment_factor=0.5,
                         perception_radius=20, max_force=1.0,
                         overcrowding_threshold=10, overcrowding_limit=20):
    """
    Analyze park capacity by running simulations with different agent counts.
    
    Args:
        space_data: Preprocessed space data
        min_agents: Minimum number of agents to test
        max_agents: Maximum number of agents to test
        step_size: Increment between agent counts
        simulation_steps: Number of steps per simulation
        overcrowding_limit: Percentage of overcrowded agents that indicates overcrowding
    
    Returns:
        Dictionary with capacity analysis results
    """
    
    print("🐕 Starting Dog Park Capacity Analysis...")
    print(f"Testing agent counts from {min_agents} to {max_agents} (step: {step_size})")
    print(f"Overcrowding threshold: {overcrowding_threshold} neighbors")
    print(f"Overcrowding limit: {overcrowding_limit}% of agents")
    print("=" * 60)
    
    agent_counts = range(min_agents, max_agents + 1, step_size)
    results = []
    
    for agent_count in agent_counts:
        print(f"\n🔄 Testing {agent_count} agents...")
        
        # Create simulation with fixed number of agents (no arrivals)
        sim = FlockingDogParkSimulation(
            space_data=space_data,
            cohesion_factor=cohesion_factor,
            separation_factor=separation_factor,
            alignment_factor=alignment_factor,
            visual_range=perception_radius,
            overcrowding_threshold=overcrowding_threshold
        )
        sim.arrival_rate = 0.0  # No new arrivals during analysis
        sim.max_force = max_force
        
        # Add the target number of agents
        sim.add_random_agents(agent_count)
        
        # Run simulation for stabilization
        stabilization_steps = min(200, simulation_steps // 2)
        for _ in range(stabilization_steps):
            sim.step()
        
        # Collect metrics during analysis period
        analysis_metrics = []
        for step in range(simulation_steps - stabilization_steps):
            sim.step()
            metrics = sim.calculate_overcrowding_metrics()
            analysis_metrics.append(metrics)
        
        # Calculate summary statistics
        if analysis_metrics:
            avg_overcrowding_pct = np.mean([m['overcrowding_percentage'] for m in analysis_metrics])
            max_overcrowding_pct = max([m['overcrowding_percentage'] for m in analysis_metrics])
            avg_neighbors = np.mean([m['average_neighbors'] for m in analysis_metrics])
            max_neighbors = max([m['max_neighbors'] for m in analysis_metrics])
            avg_sustained_overcrowding = np.mean([m['agents_with_sustained_overcrowding'] for m in analysis_metrics])
        else:
            avg_overcrowding_pct = 0
            max_overcrowding_pct = 0
            avg_neighbors = 0
            max_neighbors = 0
            avg_sustained_overcrowding = 0
        
        result = {
            'agent_count': agent_count,
            'avg_overcrowding_percentage': avg_overcrowding_pct,
            'peak_overcrowding_percentage': max_overcrowding_pct,
            'avg_neighbors': avg_neighbors,
            'max_neighbors': max_neighbors,
            'avg_sustained_overcrowding': avg_sustained_overcrowding,
            'is_overcrowded': avg_overcrowding_pct > overcrowding_limit,
            'actual_agent_count': len(sim.agents)  # Final count after simulation
        }
        
        results.append(result)
        
        # Print progress
        status = "❌ OVERCROWDED" if result['is_overcrowded'] else "✅ OK"
        print(f"   Avg Overcrowding: {avg_overcrowding_pct:.1f}% | "
              f"Peak: {max_overcrowding_pct:.1f}% | "
              f"Avg Neighbors: {avg_neighbors:.1f} | "
              f"Status: {status}")
    
    # Find optimal capacity (largest agent count before overcrowding)
    optimal_capacity = None
    for result in results:
        if not result['is_overcrowded']:
            optimal_capacity = result['agent_count']
        else:
            break
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 CAPACITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    if optimal_capacity:
        print(f"🎯 RECOMMENDED CAPACITY: {optimal_capacity} agents")
        print(f"   (Before {overcrowding_limit}% overcrowding threshold)")
    else:
        print(f"⚠️  All tested capacities exceeded {overcrowding_limit}% overcrowding!")
        print(f"   Consider increasing space or lowering agent density.")
    
    # Find the point where overcrowding starts
    first_overcrowded = next((r for r in results if r['is_overcrowded']), None)
    if first_overcrowded:
        print(f"📈 OVERCROWDING BEGINS: ~{first_overcrowded['agent_count']} agents")
        print(f"   ({first_overcrowded['avg_overcrowding_percentage']:.1f}% avg overcrowding)")
    
    return {
        'results': results,
        'optimal_capacity': optimal_capacity,
        'overcrowding_threshold_neighbors': overcrowding_threshold,
        'overcrowding_limit_percentage': overcrowding_limit,
        'parameters': {
            'cohesion': cohesion_factor,
            'separation': separation_factor,
            'alignment': alignment_factor,
            'perception_radius': perception_radius
        }
    }


def plot_capacity_analysis(capacity_results, save_figure=True):
    """
    Plot the results of capacity analysis.
    
    Args:
        capacity_results: Results from analyze_park_capacity()
        save_figure: Whether to save the plot to figures directory
    """
    results = capacity_results['results']
    optimal_capacity = capacity_results['optimal_capacity']
    
    if not results:
        print("No results to plot!")
        return
    
    agent_counts = [r['agent_count'] for r in results]
    avg_overcrowding = [r['avg_overcrowding_percentage'] for r in results]
    peak_overcrowding = [r['peak_overcrowding_percentage'] for r in results]
    avg_neighbors = [r['avg_neighbors'] for r in results]
    max_neighbors = [r['max_neighbors'] for r in results]
    
    # Create subplot figure
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#2A2A2A')
    
    # Plot 1: Overcrowding Percentage
    ax1.plot(agent_counts, avg_overcrowding, 'o-', color='#FF6B6B', linewidth=2, label='Average')
    ax1.plot(agent_counts, peak_overcrowding, 's--', color='#FF4757', linewidth=2, label='Peak')
    ax1.axhline(y=capacity_results['overcrowding_limit_percentage'], color='red', 
                linestyle=':', alpha=0.7, label=f"{capacity_results['overcrowding_limit_percentage']}% Limit")
    if optimal_capacity:
        ax1.axvline(x=optimal_capacity, color='lime', linestyle='--', alpha=0.8, 
                   label=f'Optimal: {optimal_capacity}')
    ax1.set_xlabel('Number of Agents')
    ax1.set_ylabel('Overcrowding Percentage (%)')
    ax1.set_title('Overcrowding vs Agent Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#2A2A2A')
    
    # Plot 2: Average Neighbors
    ax2.plot(agent_counts, avg_neighbors, 'o-', color='#4ECDC4', linewidth=2, label='Average')
    ax2.plot(agent_counts, max_neighbors, 's--', color='#26D0CE', linewidth=2, label='Maximum')
    ax2.axhline(y=capacity_results['overcrowding_threshold_neighbors'], color='orange', 
                linestyle=':', alpha=0.7, label=f"Threshold: {capacity_results['overcrowding_threshold_neighbors']}")
    if optimal_capacity:
        ax2.axvline(x=optimal_capacity, color='lime', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Number of Agents')
    ax2.set_ylabel('Neighbors per Agent')
    ax2.set_title('Neighbor Count vs Agent Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#2A2A2A')
    
    # Plot 3: System Status
    status_colors = ['green' if not r['is_overcrowded'] else 'red' for r in results]
    ax3.scatter(agent_counts, avg_overcrowding, c=status_colors, s=100, alpha=0.7)
    ax3.axhline(y=capacity_results['overcrowding_limit_percentage'], color='red', 
                linestyle=':', alpha=0.7)
    if optimal_capacity:
        ax3.axvline(x=optimal_capacity, color='lime', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Number of Agents')
    ax3.set_ylabel('Avg Overcrowding (%)')
    ax3.set_title('System Status (Green=OK, Red=Overcrowded)')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#2A2A2A')
    
    # Plot 4: Capacity Summary
    ax4.axis('off')
    summary_text = f"""
CAPACITY ANALYSIS SUMMARY

Overcrowding Threshold: {capacity_results['overcrowding_threshold_neighbors']} neighbors
Overcrowding Limit: {capacity_results['overcrowding_limit_percentage']}% of agents

Recommended Capacity: {optimal_capacity if optimal_capacity else 'Not found'} agents

Parameters:
• Cohesion: {capacity_results['parameters']['cohesion']:.1f}
• Separation: {capacity_results['parameters']['separation']:.1f}
• Alignment: {capacity_results['parameters']['alignment']:.1f}
• Perception: {capacity_results['parameters']['perception_radius']}

Total Tests: {len(results)}
Agent Range: {min(agent_counts)} - {max(agent_counts)}
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontfamily='monospace',
             fontsize=11, color='white', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#2A2A2A', alpha=0.8, edgecolor='white'))
    
    plt.tight_layout()
    
    if save_figure:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(os.path.join('figures', 'capacity_analysis.png'), dpi=150, 
                   facecolor='#2A2A2A', edgecolor='none')
        print(f"\n💾 Capacity analysis plot saved to figures/capacity_analysis.png")
    
    plt.show()
    
    return fig


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
    parser.add_argument('--overcrowding_threshold', type=int, default=10, help='Number of neighbors before agent is considered overcrowded')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap instead of animation')
    parser.add_argument('--capacity_analysis', action='store_true', help='Run capacity analysis to determine optimal agent count')
    parser.add_argument('--min_agents', type=int, default=20, help='Minimum agents for capacity analysis')
    parser.add_argument('--max_agents', type=int, default=300, help='Maximum agents for capacity analysis')
    parser.add_argument('--agent_step', type=int, default=20, help='Agent count increment for capacity analysis')
    parser.add_argument('--overcrowding_limit', type=float, default=20.0, help='Overcrowding percentage limit for capacity analysis')
    
    args = parser.parse_args()
    space_data = load_preprocessed_space(args.layout)
    
    if args.capacity_analysis:
        # Run capacity analysis
        results = analyze_park_capacity(
            space_data=space_data,
            min_agents=args.min_agents,
            max_agents=args.max_agents,
            step_size=args.agent_step,
            simulation_steps=args.steps,
            cohesion_factor=args.cohesion,
            separation_factor=args.separation,
            alignment_factor=args.alignment,
            perception_radius=args.perception,
            max_force=args.max_force,
            overcrowding_threshold=args.overcrowding_threshold,
            overcrowding_limit=args.overcrowding_limit
        )
        # Plot the results
        plot_capacity_analysis(results)
    elif args.heatmap:
        generate_flocking_heatmap(
            space_data=space_data,
            arrival_rate=args.arrival_rate,
            steps=args.steps,
            cohesion_factor=args.cohesion,
            separation_factor=args.separation,
            alignment_factor=args.alignment,
            perception_radius=args.perception,
            max_force=args.max_force,
            overcrowding_threshold=args.overcrowding_threshold
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
            overcrowding_threshold=args.overcrowding_threshold
        ) 