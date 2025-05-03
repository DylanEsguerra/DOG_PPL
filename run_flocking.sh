#!/bin/bash
source venv/bin/activate

# Default values
LAYOUT_PATH="layouts/processed_space.pkl"
STEPS=10000
ARRIVAL_RATE=0.1
TEMPERATURE=0.8
COHESION=0.5
SEPARATION=0.5
ALIGNMENT=0.5
PERCEPTION=20
MAX_FORCE=1.0
GENERATE_HEATMAP=false

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --layout)
      LAYOUT_PATH="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --arrival)
      ARRIVAL_RATE="$2"
      shift 2
      ;;
    --temp)
      TEMPERATURE="$2"
      shift 2
      ;;
    --cohesion)
      COHESION="$2"
      shift 2
      ;;
    --separation)
      SEPARATION="$2"
      shift 2
      ;;
    --alignment)
      ALIGNMENT="$2"
      shift 2
      ;;
    --perception)
      PERCEPTION="$2"
      shift 2
      ;;
    --max_force)
      MAX_FORCE="$2"
      shift 2
      ;;
    --heatmap)
      GENERATE_HEATMAP=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Build the command
CMD="python src/flocking_simulation.py --layout \"$LAYOUT_PATH\" --steps $STEPS --arrival_rate $ARRIVAL_RATE --cohesion $COHESION --separation $SEPARATION --alignment $ALIGNMENT --perception $PERCEPTION --max_force $MAX_FORCE"

# Add heatmap flag and temperature if needed
if [ "$GENERATE_HEATMAP" = true ]; then
  CMD="$CMD --heatmap --temperature $TEMPERATURE"
else
  # Temperature is only used for heatmap generation
  echo "Note: Temperature parameter ($TEMPERATURE) will only be used with --heatmap option"
fi

# Print and execute the command
echo "Running: $CMD"
eval $CMD

echo "Process complete!" 