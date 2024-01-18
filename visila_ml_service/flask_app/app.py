# app.py

import base64
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
from prediction import Prediction

app = Flask(__name__)
socketio = SocketIO(app)
predict = Prediction()

processed_images = []
# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

@app.route('/')
def index():
    return render_template('index.html', processed_images=processed_images)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('processed_images', processed_images)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('predict_sign')
def predict_sign(data):
    try:
        image_data = base64.b64decode(data)

        image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)

        processed_img, word_result = predict.process_image(image_np)

        _, encoded_image = cv2.imencode('.png', processed_img)
        encoded_image_data = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

        socketio.emit('processed_image', encoded_image_data)

        print(word_result)

        socketio.emit('prediction_result', word_result)
    except Exception as e:
        print(f"Error processing image: {e}")


@socketio.on('process_image')
def process_image(data):
    try:
        
        image_data = base64.b64decode(data)

        image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
        
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        processed_img, word_result = predict.process_image(image_np)

        socketio.emit('prediction_result', word_result)

        _, encoded_image = cv2.imencode('.png', processed_img)
        encoded_image_data = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

        socketio.emit('processed_image', encoded_image_data)

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True)