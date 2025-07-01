#!/usr/bin/env python3
"""
Example script for running dog park capacity analysis.

This script demonstrates how to use the new overcrowding analysis features
to determine optimal park capacity.
"""

import sys
import os
sys.path.append('src')

from flocking_simulation import (
    load_preprocessed_space, 
    analyze_park_capacity, 
    plot_capacity_analysis
)

def main():
    """Run capacity analysis example."""
    
    # Load your preprocessed layout
    layout_path = "layouts/Simple_Layout.pkl"  # Update this path
    
    if not os.path.exists(layout_path):
        print(f"❌ Layout file not found: {layout_path}")
        print("Please run the Space_Preprocessor first to create a layout file.")
        return
    
    print("🐕 Loading dog park layout...")
    space_data = load_preprocessed_space(layout_path)
    
    print("🔬 Starting capacity analysis...")
    
    # Run capacity analysis
    results = analyze_park_capacity(
        space_data=space_data,
        min_agents=25,          # Start testing at 25 agents
        max_agents=100,         # Test up to 100 agents
        step_size=25,           # Test every 25 agents
        simulation_steps=100,   # Run each test for 100 steps
        overcrowding_threshold=10,  # 10+ neighbors = overcrowded
        overcrowding_limit=15   # 15% overcrowded agents = system overcrowded
    )
    
    # Plot the results
    print("\n📈 Generating analysis plots...")
    plot_capacity_analysis(results)
    
    print("\n✅ Analysis complete!")
    print("Check the figures directory for the capacity_analysis.png plot.")

if __name__ == "__main__":
    main() 