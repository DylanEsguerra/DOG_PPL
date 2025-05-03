#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Default values
IMAGE_PATH=""
DIFFUSION_RATE=1.0
STEPS=100
ANIMATE=false
SHOW_SPACE_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --rate)
      DIFFUSION_RATE="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --animate)
      ANIMATE=true
      shift
      ;;
    --show-space)
      SHOW_SPACE_ONLY=true
      shift
      ;;
    *)
      # If no flag is provided, assume it's the image path
      if [ -z "$IMAGE_PATH" ]; then
        IMAGE_PATH="$1"
      fi
      shift
      ;;
  esac
done

# Check if image path is provided
if [ -z "$IMAGE_PATH" ]; then
  echo "Error: Image path is required"
  echo "Usage: ./run_diffusion.sh <image_path> [options]"
  echo "Options:"
  echo "  --rate <value>     Diffusion rate (default: 0.1)"
  echo "  --steps <value>    Number of simulation steps (default: 100)"
  echo "  --animate          Generate animation"
  echo "  --show-space       Only show the simulation space without running the simulation"
  exit 1
fi

# Build the command
CMD="python src/diffusion_simulation.py \"$IMAGE_PATH\" --diffusion_rate $DIFFUSION_RATE --steps $STEPS"

# Add animation flag if needed
if [ "$ANIMATE" = true ]; then
  CMD="$CMD --animate"
fi

# Add show space only flag if needed
if [ "$SHOW_SPACE_ONLY" = true ]; then
  CMD="$CMD --show_space_only"
fi

# Print the command being executed
echo "Running: $CMD"

# Execute the command
eval $CMD

echo "Process complete!"

# If we created a visualization, display it
if [ "$SHOW_SPACE_ONLY" = true ] && [ -f "simulation_space_overlay.png" ]; then
  echo "Simulation space visualization generated: simulation_space_overlay.png"
  
  # Try to open the image with the default viewer
  case "$(uname)" in
    "Darwin")
      open simulation_space_overlay.png
      ;;
    "Linux")
      xdg-open simulation_space_overlay.png
      ;;
    *)
      echo "Please open the visualization file manually: simulation_space_overlay.png"
      ;;
  esac
# If we created a heatmap, display it
elif [ -f "diffusion_heatmap.png" ]; then
  echo "Heatmap generated: diffusion_heatmap.png"
  
  # Try to open the image with the default viewer
  case "$(uname)" in
    "Darwin")
      open diffusion_heatmap.png
      ;;
    "Linux")
      xdg-open diffusion_heatmap.png
      ;;
    *)
      echo "Please open the heatmap file manually: diffusion_heatmap.png"
      ;;
  esac
fi 