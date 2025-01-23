import os
import json
from PIL import Image, ImageEnhance


def adjust_brightness_and_update_annotations(
    image_path, annotations, output_dir, new_image_id, brightness_factor
):
    try:
        # Load the image
        image = Image.open(image_path)

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        brightened_image = enhancer.enhance(brightness_factor)

        # Save the brightened image with "_augmented_brightness" appended to the filename
        brightened_image_name = (
            os.path.splitext(os.path.basename(image_path))[0]
            + "_augmented_brightness.png"
        )
        brightened_image_path = os.path.join(output_dir, brightened_image_name)
        brightened_image.save(brightened_image_path)

        # Update annotations (keeping original x, y, w, h)
        new_annotations = []
        for annotation in annotations:
            new_annotation = annotation.copy()
            new_annotation["image_id"] = new_image_id  # Update image_id to the new one
            new_annotations.append(new_annotation)

        return brightened_image_path, new_annotations

    except FileNotFoundError:
        print(f"File not found: {image_path}. Skipping this image.")
        return None, []  # Return None for the image path and empty list for annotations


def augment_coco_dataset(coco_json_path, output_dir, brightness_factor):
    # Load the COCO annotations
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the last two letters of the input JSON filename (before the extension)
    base_name = os.path.basename(coco_json_path)
    last_two_letters = os.path.splitext(base_name)[0][
        -2:
    ]  # Get the last two letters before the extension
    last_two_letters = f"_{last_two_letters}"  # Prepend an underscore

    # Create a subdirectory for images with the last two letters
    images_subdir = os.path.join(output_dir, f"images{last_two_letters}")
    os.makedirs(images_subdir, exist_ok=True)

    # Prepare new images and annotations lists
    new_images = []
    new_annotations = []

    # Process each image
    for image_info in coco_data["images"]:
        image_id = image_info["id"]
        image_path = os.path.normpath(
            os.path.join(
                os.path.dirname(coco_json_path),
                f"images{last_two_letters}",
                image_info["file_name"][7:],
            )
        ).replace("\\", "/")

        # Get annotations for the current image
        annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == image_id
        ]

        # Adjust brightness and update annotations
        new_image_id = image_id + 3100000  # Ensure unique ID for the augmented image
        brightened_image_path, updated_annotations = (
            adjust_brightness_and_update_annotations(
                image_path, annotations, images_subdir, new_image_id, brightness_factor
            )
        )
        if (
            brightened_image_path is not None
        ):  # Only add annotations if the image was found
            new_annotations.extend(updated_annotations)

            # Add new image info with 'images\\' prefix
            augmented_image_info = {
                "id": new_image_id,  # Use the updated image ID
                "file_name": f"images\\{os.path.basename(brightened_image_path)}",  # Prepend 'images\\'
                "width": 360,  # Fixed width after resizing
                "height": 240,  # Fixed height after resizing
            }
            new_images.append(augmented_image_info)
        else:
            # If the image was not found, skip adding its annotations
            print(f"Removing annotations for image ID: {image_id}")

    # Update the COCO data to only include new images and annotations
    coco_data["images"] = new_images
    coco_data["annotations"] = new_annotations

    # Save the new COCO annotations with the last two letters of the input JSON filename
    new_coco_json_path = os.path.join(
        output_dir, f"augmented_brightness{last_two_letters}.json"
    )
    with open(new_coco_json_path, "w") as f:
        json.dump(coco_data, f, indent=4)


if __name__ == "__main__":
    # Absolute path of the current file
    absolute_path = os.path.dirname(__file__)

    # Navigate to reach the project root
    project_root = os.path.abspath(os.path.join(absolute_path, "../../"))

    # Path to the input directory relative to the project root
    input_relative_path = "dataset/frames"
    input_dir_full_path = os.path.join(project_root, input_relative_path)

    # Construct the path for the input JSON file based on "last_letters"
    last_letters = "DK"  # Suffix for the JSON file
    coco_json_path = os.path.join(input_dir_full_path, f"frames_{last_letters}.json")

    # Path to the output directory relative to the project root
    output_relative_path = "dataset/augmented/augmented_brightness"
    output_dir_full_path = os.path.join(project_root, output_relative_path)

    brightness_factor = 1.2  # Example brightness factor (must be > 0)

    augment_coco_dataset(coco_json_path, output_dir_full_path, brightness_factor)
