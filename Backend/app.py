from flask import Flask, Response, render_template, jsonify, request
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
model = load_model("Backend/modelo.h5", compile=False)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Clases del modelo
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Inicializar cámara
cap = cv2.VideoCapture(0)

def generate_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Procesar la imagen con MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                # Mostrar la predicción en el frame
                cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convertir el frame a JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Generar el flujo de imágenes en tiempo real
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Renderiza tu archivo HTML

@app.route('/video')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
