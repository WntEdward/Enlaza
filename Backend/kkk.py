from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo
model = load_model("mod_rec_chido.h5", compile=False)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Clases del modelo
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recibir la imagen base64
        data = request.json.get("image")
        if data is None:
            return jsonify({"error": "No se envió imagen"}), 400

        # Decodificar la imagen
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        # Convertir a RGB y procesar con MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Extraer coordenadas y realizar predicción
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                # Convertir a numpy y realizar la predicción
                keypoints = np.array(keypoints).reshape(1, -1)
                prediction = model.predict(keypoints, verbose=0)
                class_index = np.argmax(prediction)
                class_label = class_names[class_index]
                return jsonify({"prediction": class_label})

        return jsonify({"prediction": "No se detectaron manos"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
