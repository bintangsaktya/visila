import time
import cv2
import mediapipe as mp
from cvzone.ClassificationModule import Classifier

class Prediction:
    img_size = 300
    offset = 20
    timeout_duration = 5 
    label_append_interval = 2  
    last_detection_time = time.time()
    detection_start_time = None
    word = ""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
    classifier = None
    labels = None

    def __init__(self):
        with open("./Model/labels.txt", "r") as file:
            Prediction.labels = [line.split()[1] for line in file]

        Prediction.classifier = Classifier("./Model/keras_model.h5", "./Model/labels.txt")

    @classmethod
    def get_bounding_box(cls, hand_landmarks, frame_shape):
        x = [landmark.x for landmark in hand_landmarks.landmark]
        y = [landmark.y for landmark in hand_landmarks.landmark]

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        return int(x_min * frame_shape[1]), int(y_min * frame_shape[0]), int(x_max * frame_shape[1]), int(y_max * frame_shape[0])

    @classmethod
    def process_image(cls, frame):
        current_time = time.time()
        elapsed_time = current_time - cls.last_detection_time

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get hand landmarks
        hand_landmarks_dict = cls.hands.process(frame_rgb)

        # Draw landmarks if hands are detected
        if hand_landmarks_dict.multi_hand_landmarks:
            for hand_landmarks in hand_landmarks_dict.multi_hand_landmarks:
                cls.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    cls.mp_hands.HAND_CONNECTIONS,
                    cls.mp_drawing_styles.get_default_hand_landmarks_style(),
                    cls.mp_drawing_styles.get_default_hand_connections_style(),
                )

                try:
                   # Extract bounding box coordinates
                    bBox = cls.get_bounding_box(hand_landmarks, frame.shape)

                    # Crop the image based on the bounding box
                    # t, b , l, r
                    cropped_image = frame[bBox[1] - cls.offset * 2:bBox[3] + cls.offset* 2, bBox[0] - cls.offset * 2:bBox[2] + cls.offset * 2]
                    cropped_image = cv2.resize(cropped_image, (300  , 300))
                except:
                    pass

            word_prediction = cls.predict(cropped_image=cropped_image, current_time=current_time)

        if elapsed_time > cls.timeout_duration:
            cls.detection_start_time = None
            cls.word = "" 

        return cropped_image, word_prediction

    @classmethod
    def predict(cls, cropped_image, current_time):
        try:
            cls.last_detection_time = current_time

            # Start the timer if it's the beginning of a new detection
            if cls.detection_start_time is None:
                cls.detection_start_time = current_time

            prediction, index = cls.classifier.getPrediction(cropped_image)

            print(f"Prediction: {prediction[index]}, {cls.labels[index]}")

            if current_time - cls.detection_start_time >= cls.label_append_interval and prediction[index] > 0.90:
                cls.word += cls.labels[index]
                # print(f"Word so far: {cls.word}")
                cls.detection_start_time = None
            else:
                print("Not appending")
                # print(f"Word so far: {cls.word}")

            return cls.word
        except:
            print("Error")
            pass

    @classmethod
    def get_result(cls):
        return cls.word
