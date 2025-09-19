import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('sign_language_model.pkl')
    word_dict = joblib.load('word_dictionary.pkl')
    # Invert the dictionary for prediction lookup
    inverted_word_dict = {v: k for k, v in word_dict.items()}
except FileNotFoundError:
    model = None
    inverted_word_dict = {}
    print("Warning: Model or dictionary file not found. Please run train_model.py first.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

@app.route('/sign_to_text', methods=['POST'])
def sign_to_text():
    if not model:
        return jsonify({'error': 'AI model not trained yet.'}), 500

    image_data = request.data
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid image data.'}), 400

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])
                predicted_word = inverted_word_dict.get(prediction[0], 'Unknown')
                return jsonify({'word': predicted_word})

    return jsonify({'word': 'No sign detected.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)