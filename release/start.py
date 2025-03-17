# ------------------------------------------------------------------
# Ova skripta služi za igranje igrice, kombinira slikanje slika sa laptop kamere,
# pozivanje modela neuronske mreže nad tom slikom, očitavanje predikcija, te
# mapiranje u naredbe za igicu.

# Te naredbe se zapisuju u file shared.txt, koji je smiješten u folderu sa igricom,
# a igrica od tamo čita i pomiče avion.

# Potrebno je imati dole navedene knjižnice, a ako želite igrati igricu, treba imati i unity
# bazirano na unity 2022.3.28f1. Ukoliko ne želite igricu isprobati, možete samo napraviti neki prazan
# .txt file i zaljepiti njegovu putanju.

# Sve parametre podesite u sekciji parameters. Podesite željeni FPS, putanje, i s kojom rukom igrate.


# ------------------------------------------------------------------

# Python version: 3.11.11 (main, Dec  4 2024, 08:55:07) [GCC 11.4.0]
# TensorFlow version: 2.17.1

# # TensorFlow
# pip install tensorflow==2.17.1

# # NumPy
# pip install numpy

# # Matplotlib
# pip install matplotlib

# # Scikit-learn
# pip install scikit-learn

# # OpenCV
# pip install opencv-python-headless

# # Pycocotools
# pip install pycocotools

# I MEDIA PIPE mp
# ------------------------------------------------------------------


import tensorflow as tf
import os
import json
import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
import mediapipe as mp
import math
import portalocker
import threading

####
from tensorflow.keras.metrics import BinaryAccuracy, MeanSquaredError, Metric
from tensorflow.keras import layers, models, losses, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    MaxPooling2D,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
)
from tensorflow.keras.utils import get_custom_objects

###

# Mixed precision setup (optional)
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# Checking if the versions are correct

print("\n\nTF version:", tf.__version__)
print("tf.Keras version:", tf.keras.__version__)


# --- PARAMETERS -----
absolute_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(absolute_path, "../"))

model_relative_path = "release/weights_test_model_val_0.43_240.weights.h5"
model_full_path = os.path.join(project_root, model_relative_path)
shared_relative_path = "release/Build/shared.txt"
shared_path = os.path.join(project_root, shared_relative_path)

bbox_format = "center"
interference_FPS = 2  # preporučamo 1-3
VISUALIZE_IMAGES = False  # Prikaz slika
Verbose_log = False  # Ispis informacija u terminalu
RIGHT_HAND = True  # False for LEFT HAND

EXPAND_FACTOR = 0.5
GAME_GRID = (25, 15)


# --- IMAGE PREPROCESSING ----
def downscale_image(frame, input_width, input_height):
    """
    Downscales an image to the specified dimensions using bilinear interpolation.

    ONLY 2 Supported dimension reductions:
    - from 16/9 (1920x1080) to 360x240
    - from 3/2 (1080x720) to 360x240
    """

    if (input_width / input_height) == 16 / 9:
        # print("Detected 16/9 original ratio.")
        new_width = int(input_height * 1.5)
        x_start = (input_width - new_width) // 2
        cropped_frame = frame[:, x_start : x_start + new_width]
        return cv2.resize(cropped_frame, (360, 240), interpolation=cv2.INTER_LINEAR)
    elif (input_width / input_height) == 4 / 3:
        # Calculate new width to match 16:9 ratio (i.e., crop width)
        target_width = input_width
        target_height = int(input_width * 2 / 3)

        if input_height < target_height:
            target_height = input_height
            target_width = int(input_height * 3 / 2)

        # Calculate the cropping box to center the image
        x_offset = (input_width - target_width) // 2
        y_offset = (input_height - target_height) // 2

        # Crop the image
        cropped_frame = frame[
            y_offset : y_offset + target_height, x_offset : x_offset + target_width
        ]
        return cv2.resize(cropped_frame, (360, 240), interpolation=cv2.INTER_LINEAR)
    elif (input_width / input_height) == 3 / 2:
        # print("Detected 3/2 original ratio.")
        return cv2.resize(frame, (360, 240), interpolation=cv2.INTER_LINEAR)
    else:
        print(
            f"Unsupported format -> Dimensions = {input_width}x{input_height} but has to be either 3:2 or 16:9!"
        )
        return None


def normalize_brightness_clahe(gray_frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_frame)


def compute_frame_difference_intensity(frameBEFORE, frameAFTER):
    return cv2.absdiff(frameBEFORE, frameAFTER)


def write_to_file(class_name, x_center, y_center, angle, GAME_GRID=GAME_GRID):
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(shared_path), exist_ok=True)

        # Try to write with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(shared_path, "w") as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    x = ((x_center / 360 * GAME_GRID[0] * 2) - (GAME_GRID[0])) * (-1)
                    y = ((y_center / 240 * GAME_GRID[1] * 2) - (GAME_GRID[1])) * (-1)
                    file.write(f"{class_name}\n{x}\n{y}\n{int(angle)}\n")
                    portalocker.unlock(file)
                break  # Success, exit the retry loop
            except PermissionError:
                if attempt == max_retries - 1:  # Last attempt
                    print(
                        f"Unable to write to {shared_path} after {max_retries} attempts"
                    )
                    raise
                time.sleep(0.1)  # Wait briefly before retrying
    except Exception as e:
        print(f"Error writing to file: {e}")


# --- NN ARCHITECTURE
def build_model(input_shape, grid_size=7, num_classes=4):

    input_img = Input(shape=input_shape)

    # Convolutional Backbone
    x = Conv2D(64, (7, 7), strides=2, padding="same", activation=None)(input_img)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(128, (3, 3), strides=1, padding="same", activation=None)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(64, (1, 1), strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3), strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (1, 1), strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3), strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Concatination of convolutional info
    x = Flatten()(x)

    # This Fully connected layer will be used only by bbox pred head
    x = Dense(128, activation="relu")(x)

    # This Fully connected layer will be used only by classification head
    x_class = Dense(64, activation="relu")(x)

    # Bounding box predictions
    bbox_output = Dense(4, activation="relu", name="bbox_output")(x)

    # Class probabilities
    class_output = Dense(4, activation="softmax", name="class_output")(x_class)

    # Concatenate bbox and class probabilities into a single 8-element output
    output = tf.keras.layers.Concatenate(name="output")([class_output, bbox_output])

    model = models.Model(inputs=input_img, outputs=output)

    # model.summary()
    return model


# --- LOAD MODEL --------------------------
nn_model = build_model((240, 360, 1))
# nn_model.summary()
print("NN model loaded succesfully!")

# Load the saved weights into model nn
nn_model.load_weights(model_full_path)


# --- GAME CONTOLING -----------------------------
target_x, target_y, target_angle = 0, 0, 0
current_x, current_y, current_angle = 0, 0, 0
curr_class = 0
lock = threading.Lock()


def update_raw_predictions(new_class, new_x, new_y, new_angle):
    global curr_class, target_x, target_y, target_angle
    with lock:
        curr_class, target_x, target_y, target_angle = (
            new_class,
            new_x,
            new_y,
            new_angle,
        )


def smooth_writer():
    global current_x, current_y, current_angle, curr_class
    while True:
        with lock:
            dx = target_x - current_x
            dy = target_y - current_y
            da = target_angle - current_angle
            if da > 180:
                da -= 360
            elif da < -180:
                da += 360

        # Smoothly interpolate (tweak `alpha` for speed of transition)
        alpha = 0.1
        current_x += alpha * dx
        current_y += alpha * dy
        current_angle += (alpha * da) % 360
        current_angle = current_angle % 360

        # Write to file
        write_to_file(curr_class, current_x, current_y, current_angle)

        # Adjust writing frequency
        time.sleep(1 / 30)  # 30 FPS


# Start the smooth writer thread
threading.Thread(target=smooth_writer, daemon=True).start()


# --- LIVE INTERFERENCE --------------------------

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

cam_fps = cap.get(cv2.CAP_PROP_FPS)


# Initialize MediaPipe Hands object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.1
)
last_goog_angle_deg = 0
class_history = []

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    if VISUALIZE_IMAGES:
        plt.ion()  # Turn on interactive mode for live plotting
        fig, ax = plt.subplots()

    while True:
        full_start_time = time.time()

        # Convert the frame to RGB (OpenCV uses BGR by default)
        if Verbose_log:
            print(
                f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}, time taken: {(1/cap.get(cv2.CAP_PROP_FPS)):.5f}s"
            )

        # Capture the second frame
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Could not read second frame. Skipping...")
            time.sleep(0.5)
            continue

        # AFTER diff FRAME (main frame)
        start_time = time.time()
        gray_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        height, width, channels = frame2.shape
        downscaled_grey_frame = downscale_image(gray_frame, width, height)
        normalized_down_grey_frame = normalize_brightness_clahe(downscaled_grey_frame)
        frameAFTER = normalized_down_grey_frame

        prep_t = time.time() - start_time
        if Verbose_log:
            print(f"Time to preprocess 1 frame: {prep_t:.8f} seconds")

        # NN MODEL INTERFERENCE
        # ----------------------
        frameAFTER_normalized = frameAFTER / 255.0
        frameAFTER_normalized = frameAFTER_normalized[np.newaxis, ..., np.newaxis]

        # print(nn_model.input_shape)
        # print("Shape of frameAFTER_normalized:", frameAFTER_normalized.shape)
        start_time = time.time()
        pred_output = nn_model.predict(frameAFTER_normalized, verbose=0)
        prediction_time = time.time() - start_time
        if Verbose_log:
            print("### PREDICTION TIME (s) :", prediction_time)
            print("Model NN prediction:", pred_output)
        # ----------------------

        # ANGLE CALCULATION
        angle_goog_time = 0
        CROP = False
        pred_class = np.argmax(pred_output[0][0:4])
        if pred_class != 1:
            angle_start = time.time()

            if frameAFTER.size == 0:
                print("Error: Cropped image is empty!")

            else:
                image_np = cv2.cvtColor(frameAFTER, cv2.COLOR_RGB2BGR)
                results = hands.process(image_np)

                if results.multi_hand_landmarks:
                    if Verbose_log:
                        print("/////////")
                        print("/////////")
                        print(f"GOOGLE Hands detected!")
                        print("/////////")
                        print("/////////")
                    for hand_landmarks in results.multi_hand_landmarks:
                        if RIGHT_HAND:
                            landmark_8 = hand_landmarks.landmark[20]
                            landmark_20 = hand_landmarks.landmark[8]
                        else:
                            landmark_8 = hand_landmarks.landmark[8]
                            landmark_20 = hand_landmarks.landmark[20]

                        h, w, _ = image_np.shape
                        point_8 = (int(landmark_8.x * w), int(landmark_8.y * h))
                        point_20 = (int(landmark_20.x * w), int(landmark_20.y * h))

                        dx = point_20[0] - point_8[0]
                        dy = point_20[1] - point_8[1]
                        angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(angle_rad)
                        if angle_deg < 0:
                            angle_deg += 360
                        goog_angle_deg = int(round(angle_deg))
                        last_goog_angle_deg = goog_angle_deg
                        angle_goog_time = time.time() - start_time
                        if Verbose_log:
                            print(f"Angle GOOGLE: {goog_angle_deg}°")
                            print("### ANGLE (GOOGLE) TIME (s) :", angle_goog_time)

        # MOST COMMMON CLASS
        class_history.append(pred_class)
        if len(class_history) > 3:
            class_history.pop(0)
        class_count = [class_history.count(i) for i in set(class_history)]
        most_common_class = [
            i for i in set(class_history) if class_history.count(i) == max(class_count)
        ]
        curr_class = most_common_class[0] if len(most_common_class) == 1 else 0

        write_time = 0
        if pred_class != 1 and curr_class != 1:
            # write_to_file(pred_class, pred_output[0][4], pred_output[0][5], last_goog_angle_deg)
            if curr_class == 2:
                update_raw_predictions(
                    2, pred_output[0][4], pred_output[0][5], last_goog_angle_deg
                )
            elif pred_class != 2:
                update_raw_predictions(
                    pred_class,
                    pred_output[0][4],
                    pred_output[0][5],
                    last_goog_angle_deg,
                )

        # Display the image on Matplotlib
        wait_t = (
            int(
                (1000 / interference_FPS)
                - 130
                - (3 * prep_t * 1000)
                - (prediction_time * 1000)
                - (angle_goog_time * 1000)
                - (write_time * 1000)
            )
            / 1000
        )  # in seconds # još i model predict time
        if Verbose_log:
            print("WAITTT TIMEEE:", wait_t)
        if wait_t < 0:
            wait_t = int(1000 / interference_FPS) / 1000
        if VISUALIZE_IMAGES:
            ax.clear()  # Clear the previous image
            ax.imshow(frameAFTER, cmap="gray")
            ax.text(
                10,
                10,
                f"Class: {pred_class}, angl: {last_goog_angle_deg}",
                color="white",
                fontsize=12,
                ha="left",
                va="top",
            )
            # ax.axis('off')  # Turn off axis
            plt.pause(0.00001)
        time.sleep(wait_t)

        # Calculate FPS
        elapsed_time = time.time() - full_start_time
        aprox_max_time = (
            100
            + (2.5 * prep_t * 1000)
            + (prediction_time * 1000)
            + (angle_goog_time * 1000)
            + (write_time * 1000)
        ) / 1000
        if Verbose_log:
            print(
                f"Game FPS: {(1 / elapsed_time):.2f}, time taken: {elapsed_time:.5f}s"
            )
            print(
                f"MAX game FPS: {(1 / aprox_max_time):.2f}, time taken: {aprox_max_time:.5f}s"
            )  # when no ploting img
            print("--------------")
            print()

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode
    plt.show()

write_to_file(0, 180, 120, 0)
