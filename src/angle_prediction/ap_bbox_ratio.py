import os
import json

def calculate_angle(width, height):
    """Calculate the angle based on the aspect ratio and clamp the value."""
    ratio = width / height
    
    # Interpolate between 0 and 90 for ratios between 3:1 and 1:1
    angle = (ratio - 1) * (90 - 0) / (3 - 1)

    # Clamp the angle to the range [0, 90]
    angle = max(0.0, min(angle, 90.0))

    # Shift the angle to the range [-45, 45]
    return angle - 45

def parseData(coco_json_path):
    with open(coco_json_path, 'r') as file:
        data = json.load(file)

    # Create a mapping of category IDs to names
    category_map = {category['id']: category['name'] for category in data['categories']}

    for annotation in data['annotations']:
        bbox = annotation['bbox']
        width = bbox[2]
        height = bbox[3]
        angle = calculate_angle(width, height)

        category_name = category_map.get(annotation['category_id'], "Unknown")
        
        output = {
            "id": annotation['id'],
            "image_id": annotation['image_id'],
            "category_name": category_name,
        }

        if category_name == "FLY":
            # Update the output with the predicted angle for "FLY" category
            output.update({"predicted_angle": round(angle, 2)})

        print(json.dumps(output))  # Print as valid JSON

if __name__ == "__main__":
    input_path = "C:/Users/Skull Mini/Downloads/neumre_projekt/original"  # Your path
    last_letters = "JJ"  # Last letters to use
    coco_json_path = os.path.join(input_path, f"result_{last_letters}.json")
    parseData(coco_json_path)
