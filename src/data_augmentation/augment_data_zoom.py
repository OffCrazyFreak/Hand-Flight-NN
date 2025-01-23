import os
import json
from PIL import Image


def zoom_image_and_update_annotations(
    image_path, annotations, output_dir, new_image_id, zoom_factor
):
    try:
        # Load the image
        image = Image.open(image_path)
        width, height = image.size

        # Calculate the cropping box for the center of the image
        crop_width, crop_height = 360 // zoom_factor, 240 // zoom_factor
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image to the center
        cropped_image = image.crop((left, top, right, bottom))

        # Resize (zoom) the cropped image back to 360x240
        zoomed_image = cropped_image.resize((360, 240), Image.LANCZOS)

        # Save the zoomed image with "_augmented_zoom" appended to the filename
        zoomed_image_name = (
            os.path.splitext(os.path.basename(image_path))[0] + "_augmented_zoom.png"
        )
        zoomed_image_path = os.path.join(output_dir, zoomed_image_name)
        zoomed_image.save(zoomed_image_path)

        # Update annotations (keeping original x, y, w, h)
        new_annotations = []

        for annotation in annotations:

            x, y, w, h = annotation["bbox"]

            if x <= left or y <= top or x >= right or y >= bottom:
                continue
            else:
                new_annotation = annotation.copy()

                # Calculate new coordinates relative to the cropped area
                new_x = max(0, x - left)  # Adjust x to be relative to the crop
                new_y = max(0, y - top)  # Adjust y to be relative to the crop
                new_w = min(w, right - x)  # Adjust width to fit within the crop
                new_h = min(h, bottom - y)  # Adjust height to fit within the crop

                # Scale the bounding box to the new image size (360x240)
                new_annotation["bbox"] = [
                    new_x * zoom_factor,  # Scale x
                    new_y * zoom_factor,  # Scale y
                    new_w * zoom_factor,  # Scale width
                    new_h * zoom_factor,  # Scale height
                ]
                new_annotation["image_id"] = (
                    new_image_id  # Update image_id to the new one
                )
                new_annotations.append(new_annotation)

        return zoomed_image_path, new_annotations

    except FileNotFoundError:
        print(f"File not found: {image_path}. Skipping this image.")
        return None, []  # Return None for the image path and empty list for annotations


def augment_coco_dataset(coco_json_path, output_dir, zoom_factor):
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

        # Zoom image and update annotations
        new_image_id = image_id + 2300000  # Ensure unique ID for the zoomed image
        zoomed_image_path, updated_annotations = zoom_image_and_update_annotations(
            image_path, annotations, images_subdir, new_image_id, zoom_factor
        )
        if zoomed_image_path is not None:  # Only add annotations if the image was found
            new_annotations.extend(updated_annotations)

            # Add new image info with 'images\\' prefix
            zoomed_image_info = {
                "id": new_image_id,  # Use the updated image ID
                "file_name": f"images\\{os.path.basename(zoomed_image_path)}",  # Prepend 'images\\'
                "width": 360,  # Fixed width after resizing
                "height": 240,  # Fixed height after resizing
            }
            new_images.append(zoomed_image_info)
        else:
            # If the image was not found, skip adding its annotations
            print(f"Removing annotations for image ID: {image_id}")

    # Update the COCO data to only include new images and annotations
    coco_data["images"] = new_images
    coco_data["annotations"] = new_annotations

    # Save the new COCO annotations with the last two letters of the input JSON filename
    new_coco_json_path = os.path.join(
        output_dir, f"augmented_zoom{last_two_letters}.json"
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
    output_relative_path = "dataset/augmented/augmented_zoom"
    output_dir_full_path = os.path.join(project_root, output_relative_path)

    zoom_factor = 1.7 # Example zoom factor

    augment_coco_dataset(coco_json_path, output_dir_full_path, zoom_factor)
