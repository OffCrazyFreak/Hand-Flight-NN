import cv2 
import os
import time
import random
import string

# PROMIJENITE
# Absolute path of the current file
absolute_path = os.path.dirname(__file__)

# Navigate to reach the project root
project_root = os.path.abspath(os.path.join(absolute_path, "../"))

recording_relative_path = "dataset/recordings/RickRoll_example.mp4"  # path od snimke
frames_relative_path = "dataset/frames/RickRoll" # path do foldera gdje ce se spremiti slike
movement_relative_path = "dataset/movement/RickRoll" # path do foldera gdje ce se spremiti slike pokreta

# Normalize the constructed path
recording_path = os.path.normpath(os.path.join(project_root, recording_relative_path))
frames_path = os.path.normpath(os.path.join(project_root, frames_relative_path))
movement_path = os.path.normpath(os.path.join(project_root, movement_relative_path))

os.makedirs(frames_path, exist_ok=True)
os.makedirs(movement_path, exist_ok=True)

print("Recording path:", recording_path)
print("Frames path:", frames_path)
print("Movement path:", movement_path)

# Adjust this to control how many frames you want per second of the video
frames_per_second = 2  

# ako False, onda možete runat kod bez da se stvarno spremaju frameovi
# tako možete vidjeti kako radi, radi li sve, i saznati info poput FPSa vaše kamere, dimenzije i trajanje.
# promjeni u True kad ste spremni spremiti slike na disk
doSAVE = True


def downscale_image(frame, input_width, input_height):
    """
    Downscales an image to the specified dimensions using bilinear interpolation.

    ONLY 2 Supported dimension reductions:
    - from 16/9 (1920x1080) to 360x240
    - from 3/2 (1080x720) to 360x240

    :param frame: Input image to be downscaled.
    :param input_width: Width of the input image.
    :param input_height: Height of the input image.
    :return: Downscaled image or None if the aspect ratio is unsupported.
    """
    if (input_width / input_height) == 16 / 9:
        # print("Detected 16/9 original ratio.")
        new_width = int(input_height * 1.5)
        x_start = (input_width - new_width) // 2
        cropped_frame = frame[:, x_start:x_start + new_width]
        return cv2.resize(cropped_frame, (360, 240), interpolation=cv2.INTER_LINEAR)
    elif (input_width / input_height) == 3 / 2:
        # print("Detected 3/2 original ratio.")
        return cv2.resize(frame, (360, 240), interpolation=cv2.INTER_LINEAR)
    else:
        print(f"Unsupported format -> Dimensions = {input_width}x{input_height}")
        return None

def normalize_brightness_clahe(gray_frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Normalize brightness using adaptive histogram equalization (CLAHE).

    :param gray_frame: Grayscale input image.
    :param clip_limit: Threshold for contrast limiting.
    :param tile_grid_size: Size of grid for the CLAHE algorithm.
    :return: Brightness-normalized image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_frame)

def compute_frame_difference_intensity(frameBEFORE, frameAFTER):
    """
    Computes the absolute difference between two frames to highlight motion.

    :param frameBEFORE: The earlier frame.
    :param frameAFTER: The later frame.
    :return: Absolute difference image showing motion intensity.
    """
    return cv2.absdiff(frameBEFORE, frameAFTER)


# Generate a unique session ID for saving frames
letters = ''.join(random.choices(string.ascii_uppercase, k=3))
digits = ''.join(random.choices(string.digits, k=3))
session_id = f"{letters}{digits}"

# Load the video
cap = cv2.VideoCapture(recording_path)

if not cap.isOpened():
    print("Cannot open video. Check the path!")
    exit()

# Extract video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

# Calculate the interval between frames to match the desired frames per second
SVAKI_Nti_FRAME = max(1, int(fps // frames_per_second))  # Ensure at least one frame per second
NEIGHBOUR_dist = int(round((fps * 0.1) - 0.1))  # Distance between frames for motion calculation

# Frame processing variables
frame_number = 0
saved_count = 0
total_saved_frames = 0
frameBEFORE = None


while True:
    ret, frame = cap.read()
    if not ret:  # No more frames
        break

    frame_number += 1

    if (frame_number % SVAKI_Nti_FRAME == 0):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downscale the frame
        height, width, channels = frame.shape
        downscaled_grey_frame = downscale_image(gray_frame, width, height)
        if downscaled_grey_frame is None:
            continue

        # Normalize brightness
        normalized_down_grey_frame = normalize_brightness_clahe(downscaled_grey_frame)

        # Compute motion difference if a previous frame exists
        diff_frame_intensity = None  # Initialize diff_frame_intensity
        if frameBEFORE is not None:
            diff_frame_intensity = compute_frame_difference_intensity(frameBEFORE, normalized_down_grey_frame)

        frameBEFORE = normalized_down_grey_frame

        # print(f"Processed frame {frame_number}: Dimensions = {width}x{height}, FPS = {fps}")

        # Save frames if required
        if doSAVE:
            frame_file = os.path.join(frames_path, f"{session_id}_frame{frame_number}.png")
            cv2.imwrite(frame_file, normalized_down_grey_frame)

            if diff_frame_intensity is not None:  # Only save movement frame if it's defined
                movement_file = os.path.join(movement_path, f"{session_id}_frame{frame_number}_movement.png")
                cv2.imwrite(movement_file, diff_frame_intensity)

        total_saved_frames += 1


# Summary and cleanup
print()
print("---------------")
print("Session ID:", session_id)

print(f"\nVideo Duration: {duration:.2f} seconds")
print(f"FPS: {fps}")
print("Frames Per Second for Extraction:", frames_per_second)

print("\nTotal frames:", frame_number)
if doSAVE:
    print("Total saved frames:", total_saved_frames)
    #print("SVAKI_Nti_FRAME =", SVAKI_Nti_FRAME)

cap.release()
cv2.destroyAllWindows()