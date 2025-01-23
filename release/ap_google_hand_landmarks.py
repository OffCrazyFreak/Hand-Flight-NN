import cv2
import mediapipe as mp
import math

# Racuna nagib ruke pomocu polozaja palca i malog prsta.
# Prima string, path do slike
# Vraca nagib u stupnjevima (float) i sliku s oznacenim landmarkovima (samo za provjeru, mozemo to maknut ako necemo koristit)
def calculate_hand_tilt(image_path):

    # inicijaliziraj MediaPipe Hand Landmakrer
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # detection confidence je nizak jer dosta lose pronalazi slike dlanova koji su prstima prema kameri
    # s obzirom da mu dajemo slike na kojima su sigurno dlanovi, mislim da je nizak confidence okej
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1) as hands:
        results = hands.process(image)

        if not results.multi_hand_landmarks:
            print("No hands detected!")
            return None, image

        # procesiraj ruku
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # nacrtaj landmarkove
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # koordinate vrha palca i vrha malog prsta
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # pretvori normalizirane u pixel koordinate
        h, w, _ = image.shape  # dimenzije slike
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        pinky_x, pinky_y = int(pinky_tip.x * w), int(pinky_tip.y * h)

        # vektor od palca do malog prsta
        vector_x = thumb_x - pinky_x
        vector_y = thumb_y - pinky_y

        # kut s obzirom na vertikalnu os
        angle_radians = math.atan2(vector_y, vector_x)  # u radijanima
        angle_degrees = math.degrees(angle_radians)     # u stupnjevima

        return angle_degrees, image