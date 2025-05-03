import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simulation import DogParkSimulation
import argparse

def generate_heatmap(width=100, height=100, obstacle_size=20, 
                    arrival_rate=0.1, temperature=1.0, steps=1000):
    """Generate a heatmap of dog density over the simulation."""
    
    # Create and run the simulation
    sim = DogParkSimulation(width, height, obstacle_size, arrival_rate, temperature)
    
    # Initialize a density grid
    density_grid = np.zeros((height, width))
    
    # Run the simulation, tracking positions
    for _ in range(steps):
        sim.step()
        
        # Record current dog positions in the density grid
        for dog in sim.dogs:
            x, y = int(dog[0]), int(dog[1])
            if 0 <= x < width and 0 <= y < height:
                density_grid[y, x] += 1
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the heatmap
    mask = np.zeros_like(density_grid, dtype=bool)
    # Mask the obstacle area
    obstacle_x, obstacle_y = sim.obstacle_x, sim.obstacle_y
    mask[obstacle_y:obstacle_y+obstacle_size, obstacle_x:obstacle_x+obstacle_size] = True
    
    # Apply mask to density grid
    masked_grid = np.ma.array(density_grid, mask=mask)
    
    # Plot the heatmap
    heatmap = ax.imshow(masked_grid, cmap='hot', interpolation='nearest')
    
    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Dog Density (visits per cell)')
    
    # Add the obstacle outline
    obstacle = patches.Rectangle(
        (obstacle_x, obstacle_y), 
        obstacle_size, 
        obstacle_size, 
        linewidth=2, 
        edgecolor='black', 
        facecolor='none',
        label='Obstacle'
    )
    ax.add_patch(obstacle)
    
    # Add the entry point
    entry = patches.Circle(
        (sim.entry_x, sim.entry_y), 
        radius=2, 
        edgecolor='green', 
        facecolor='green',
        label='Entry Point'
    )
    ax.add_patch(entry)
    
    # Add title and labels
    ax.set_title(f'Dog Density Heatmap (Arrival Rate: {arrival_rate}, Temperature: {temperature})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Add a legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return density_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a dog density heatmap')
    parser.add_argument('--width', type=int, default=100, help='Width of the park')
    parser.add_argument('--height', type=int, default=100, help='Height of the park')
    parser.add_argument('--obstacle_size', type=int, default=20, help='Size of the obstacle')
    parser.add_argument('--arrival_rate', type=float, default=0.1, help='Dog arrival rate (0-1)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter (0-5)')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
    
    args = parser.parse_args()
    
    density_grid = generate_heatmap(
        width=args.width,
        height=args.height,
        obstacle_size=args.obstacle_size,
        arrival_rate=args.arrival_rate,
        temperature=args.temperature,
        steps=args.steps
    ) 