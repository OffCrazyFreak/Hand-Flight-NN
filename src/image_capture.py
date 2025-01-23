import cv2
import os
import time
from datetime import datetime

def downscale_and_grayscale(frame):
    """Downscale the image to 360x240 and convert to grayscale."""
    # Downscale the image
    downscaled_frame = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_LINEAR)
    # Convert to grayscale
    gray_frame = cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

def start_feed(output_folder, capture_rate):
    """Function to start capturing images from the camera."""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Calculate the time interval between captures
    interval = 1.0 / capture_rate

    captured_images = 0

    while True:
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame
        processed_frame = downscale_and_grayscale(frame)

        # Get the current time for the filename
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        # Save the processed frame with the new filename format
        image_filename = os.path.join(output_folder, f"camera_feed_{current_time}_{captured_images + 1}.png")
        cv2.imwrite(image_filename, processed_frame)
        print(f"Saved: {image_filename}")

        captured_images += 1

        # Wait for the remaining time to maintain the capture rate
        elapsed_time = time.time() - start_time
        time_to_wait = interval - elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Display the processed frame (optional)
        cv2.imshow("Camera Feed", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Captured {captured_images} images.")
            break

    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Finished capturing images.")

if __name__ == "__main__":
    output_folder = "/home/snow/Downloads/camera_feed"  # Folder to save images
    capture_rate = 3  # Number of images to capture per second

    start_feed(output_folder, capture_rate)
