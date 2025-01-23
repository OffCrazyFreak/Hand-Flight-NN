import os
import json
from tkinter import Tk, Label, Button, Canvas, Toplevel, Text, Scrollbar, Frame
from PIL import Image, ImageTk


class ImageAnnotationViewer:
    def __init__(self, root, output_dir, last_letters, json_name):
        self.root = root
        self.root.title("Image Annotation Viewer")

        self.last_letters = last_letters
        self.output_dir = output_dir
        self.images_dir = os.path.join(self.output_dir, f"images_{self.last_letters}")
        self.json_path = os.path.join(
            self.output_dir, f"{json_name}_{last_letters}.json"
        )

        self.images = []
        self.annotations = {}
        self.current_index = 0
        self.categories = {}

        # Load images and annotations
        self.load_data()

        # Label to display the file path and name
        self.path_info_label = Label(root, text="", font=("Arial", 10))
        self.path_info_label.pack()

        self.file_info_label = Label(root, text="", font=("Arial", 10))
        self.file_info_label.pack()

        # Create GUI components
        self.canvas = Canvas(root, width=360, height=240)
        self.canvas.pack()

        self.label = Label(root, text="")
        self.label.pack()

        self.prev_button = Button(root, text="Previous", command=self.show_prev)
        self.prev_button.pack(side="left")

        self.next_button = Button(root, text="Next", command=self.show_next)
        self.next_button.pack(side="right")

        self.full_annotation_button = Button(
            root, text="Show Full Annotation", command=self.show_full_annotation
        )
        self.full_annotation_button.pack(side="bottom")

        # Show the first image
        self.show_image()

    def load_data(self):
        # Load the augmented JSON file
        with open(self.json_path, "r") as f:
            coco_data = json.load(f)

        # Extract images and annotations
        for image_info in coco_data["images"]:
            self.images.append(image_info["file_name"])
            self.annotations[image_info["file_name"]] = [
                ann
                for ann in coco_data["annotations"]
                if ann["image_id"] == image_info["id"]
            ]

        # Load categories for displaying names
        for category in coco_data["categories"]:
            self.categories[category["id"]] = category["name"]

    def show_image(self):
        if self.current_index < 0 or self.current_index >= len(self.images):
            return

        image_file = self.images[self.current_index]
        image_path = os.path.join(self.images_dir, image_file[7:])

        # Load and display the image
        try:
            # Load and display the image
            img = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image file not found: {image_path}. Skipping this image.")
            self.current_index += 1  # Move to the next image
            if self.current_index < len(self.images):
                self.show_image()  # Show the next image
            return

        img = img.resize((360, 240), Image.LANCZOS)  # Use LANCZOS for resizing
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Update the file path and name label
        self.path_info_label.config(text=f"Path: {self.images_dir}")
        self.file_info_label.config(text=f"File: {image_file[7:]}")

        # Clear previous bounding boxes
        self.canvas.delete("bbox")  # Remove any existing bounding boxes

        # Draw bounding boxes
        for annotation in self.annotations[image_file]:
            bbox = annotation["bbox"]
            x, y, w, h = bbox
            # Draw the rectangle (x, y, x + w, y + h)
            self.canvas.create_rectangle(
                x * 360 / img.width,  # Scale x to canvas size
                y * 240 / img.height,  # Scale y to canvas size
                (x + w) * 360 / img.width,  # Scale width to canvas size
                (y + h) * 240 / img.height,  # Scale height to canvas size
                outline="red",  # Color of the bounding box
                width=2,  # Width of the bounding box line
                tags="bbox",  # Tag for easy deletion
            )

        # Display category names
        category_names = [
            self.categories[ann["category_id"]] for ann in self.annotations[image_file]
        ]
        annotation_text = f"Categories: {', '.join(category_names)}"
        self.label.config(text=annotation_text)

    def show_next(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image()

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_full_annotation(self):
        # Create a new window to display the full annotation
        annotation_window = Toplevel(self.root)
        annotation_window.title("Full Annotation")

        # Create a text area with a scrollbar
        text_frame = Frame(annotation_window)
        text_frame.pack(fill="both", expand=True)

        text_area = Text(text_frame, wrap="word")
        text_area.pack(side="left", fill="both", expand=True)

        scrollbar = Scrollbar(text_frame, command=text_area.yview)
        scrollbar.pack(side="right", fill="y")

        text_area.config(yscrollcommand=scrollbar.set)

        # Get the full annotation for the current image
        image_file = self.images[self.current_index]
        full_annotation = json.dumps(self.annotations[image_file], indent=4)

        # Insert the full annotation into the text area
        text_area.insert("1.0", full_annotation)
        text_area.config(state="disabled")  # Make the text area read-only


if __name__ == "__main__":
    absolute_path = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(absolute_path, "../"))
    relative_path = "dataset/augmented/augmented_horizontal_flip"
    images_full_path = os.path.join(project_root, relative_path)
    
    json_name = "augmented_horizontal_flip" # "frames" or "augmented_zoom" ot "augmented_brightness" or "augmented_noise" or "augmented_horizontal_flip"
    last_letters = "DK"  # Suffix for the result file

    root = Tk()

    viewer = ImageAnnotationViewer(root, images_full_path, last_letters, json_name)
    
    root.mainloop()
