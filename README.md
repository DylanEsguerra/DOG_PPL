# Dog Park Simulation with JSON Layout

This project implements a flocking behavior simulation for dogs in a custom-defined park layout. It uses the Boids algorithm to simulate realistic social behavior of dogs.

## New JSON Layout System

The new JSON-based layout system allows for precise definition of:
- Custom park boundaries of any shape
- Multiple entry points for dogs
- Various obstacles (rectangles, circles, polygons)
- Special zones for analysis

### How to Run with a JSON Layout

```bash
python src/json_layout.py layouts/rooftop_garden.json
```

With options:
```bash
python src/json_layout.py layouts/rooftop_garden.json --width 1200 --arrival_rate 0.2 --temperature 1.0 --cohesion 0.6 --separation 0.4 --perception 15 --steps 2000 --heatmap
```

## Blueprint to JSON Conversion

To create a JSON layout from a blueprint image:

```bash
python src/blueprint_to_json.py path/to/blueprint.jpg --name "My Layout" --output layouts/my_layout.json
```

This tool extracts the boundary from a blueprint image (looking for red outlines by default) and creates a basic JSON layout file.

Options:
- `--color` - Specify the color to detect (red, blue, green)
- `--no-preview` - Skip showing the preview of the extracted boundary
- `--output` - Specify the output JSON file path
- `--name` - Set the name of the layout

## Interactive Layout Editor

After creating a basic layout from a blueprint, you can use the layout editor to add zones, obstacles, and entry points:

```bash
python src/layout_editor.py layouts/my_layout.json --background path/to/blueprint.jpg
```

The editor provides a visual interface for:
- Adding zones (for analysis)
- Adding obstacles (rectangles, circles, polygons)
- Adding entry points
- Saving the completed layout

### Usage:
1. Click the appropriate button (Add Zone, Add Obstacle, Add Entry)
2. Select a shape type (Rectangle, Circle, Polygon)
3. Click on the canvas to place points
4. Press Enter to complete the shape
5. Click Save to save your layout

### JSON Layout Format

The layout file uses the following format:

```json
{
    "name": "Layout Name",
    "width": 1000,
    "height": 1000,
    "boundary": [
        [x1, y1],
        [x2, y2],
        ...
    ],
    "entry_points": [
        {
            "name": "Entry Point Name",
            "position": [x, y],
            "properties": {
                "type": "main"
            }
        }
    ],
    "obstacles": [
        {
            "name": "Obstacle Name",
            "shape": "rectangle",
            "position": [x, y],
            "width": w,
            "height": h,
            "properties": {
                "type": "obstacle_type"
            }
        },
        {
            "name": "Circle Obstacle",
            "shape": "circle",
            "position": [x, y],
            "radius": r,
            "properties": {
                "type": "circle_type"
            }
        },
        {
            "name": "Polygon Obstacle",
            "shape": "polygon",
            "points": [
                [x1, y1],
                [x2, y2],
                ...
            ],
            "properties": {
                "type": "polygon_type"
            }
        }
    ],
    "zones": [
        {
            "id": "zone_id",
            "name": "Zone Name",
            "points": [
                [x1, y1],
                [x2, y2],
                ...
            ],
            "properties": {
                "type": "zone_type",
                "label": "Display Label",
                "color": "#HEX_COLOR",
                "alpha": 0.3
            }
        }
    ]
}
```

### Layout Definition Details

#### Boundary
The `boundary` defines the outer shape of the park as a polygon. All coordinates should be in the range [0, width] and [0, height].

#### Entry Points
`entry_points` define where dogs enter the simulation. You can specify multiple entry points, and dogs will randomly choose one when they arrive.

#### Obstacles
`obstacles` can be rectangles, circles, or polygons. Dogs will avoid these areas. Each obstacle type has specific properties:
- Rectangle: `position`, `width`, `height`
- Circle: `position`, `radius`
- Polygon: `points`

#### Zones
`zones` are special areas used for analysis. They don't affect movement but can track how many dogs are in each zone during the simulation.

## Workflow for Creating a Custom Layout

1. **Extract boundary from blueprint**: 
   ```bash
   python src/blueprint_to_json.py blueprint.jpg --output layouts/initial_layout.json
   ```

2. **Edit layout with visual tool**:
   ```bash
   python src/layout_editor.py layouts/initial_layout.json --background blueprint.jpg
   ```

3. **Run simulation**:
   ```bash
   python src/json_layout.py layouts/initial_layout.json
   ```

4. **Generate heatmap**:
   ```bash
   python src/json_layout.py layouts/initial_layout.json --heatmap
   ```

## Example Layouts

The project includes sample layouts in the `layouts` directory:
- `rooftop_garden.json`: A rooftop garden with multiple zones and obstacles

## Extending the System

To add new features:
1. Define new properties in the JSON format
2. Extend the JSONLayoutSimulation class to handle the new properties
3. Update visualization functions to display the new elements

## Additional Features

- **Analytics**: Track dog movement patterns within different zones
- **Multiple Species**: Add different types of agents with varying behaviors
- **Time-based Events**: Schedule events like temporary obstacles or changing behavior parameters

## Features
- 2D simulation of dog movement in a park space
- Random walker movement model for dogs
- Flocking behavior for more realistic movement
- Custom layout support using blueprint images
- Configurable arrival rate and movement temperature
- Visualization of dog movement and density

## Setup
1. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```
2. Install dependencies (if not already installed):
   ```
   ./run.sh install
   ```
3. Run the simulation:
   ```
   ./run.sh simulation
   ```

## Simulation Types

### Basic Random Walker
```
./run.sh simulation [options]
./run.sh heatmap [options]
```

### Flocking Behavior
```
./run.sh flocking [options]
./run.sh flocking-heatmap [options]
```

### Custom Layout
To use a custom park layout from a blueprint:
1. Place your blueprint image in the `data/` directory
2. The red boundary line in the image defines the park space
3. Run the simulation with:
```
./run.sh custom data/your_blueprint.jpg [options]
./run.sh custom-heatmap data/your_blueprint.jpg [options]
```

## Parameters
- `--width`, `--height`: Dimensions of the simulation space
- `--arrival_rate`: Rate at which dogs arrive (0-1)
- `--temperature`: Controls randomness of movement (0-5)
- `--steps`: Number of simulation steps

### Flocking Parameters
- `--cohesion`: How much dogs are attracted to each other (0-1)
- `--separation`: How much dogs avoid each other (0-1)
- `--perception`: How far dogs can sense other dogs

### Custom Layout Parameters
- `--entry_x`, `--entry_y`: Entry point coordinates (0-1)

## Future Plans
- Web app with interactive sliders for parameters 