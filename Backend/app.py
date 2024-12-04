import os
import sys
import time
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2

# Verificar si el script se está ejecutando como un archivo .exe empaquetado por PyInstaller
if getattr(sys, 'frozen', False):
    model_path = os.path.join(sys._MEIPASS, 'modelo.h5')  # Ruta para el modelo cuando el archivo está empaquetado
    hand_landmark_path = os.path.join(sys._MEIPASS, 'mediapipe')  # Ruta para los archivos de mediapipe
else:
    model_path = os.path.join(os.getcwd(), 'Backend', 'modelo.h5')  # Ruta local cuando el script no está empaquetado
    hand_landmark_path = 'mediapipe'  # Ruta local para mediapipe

# Inicializar Flask
app = Flask(__name__)

# Intentar cargar el modelo de Keras y manejar el error si el formato es incorrecto
try:
    model = load_model(model_path, compile=False)
    print("Modelo cargado correctamente.")
except ValueError as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)  # Terminar el script si el modelo no se puede cargar

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo manual de las clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Variables para la predicción de palabras
current_word = ""  # Palabra que se está formando
last_letter = ""  # Última letra detectada para evitar duplicados consecutivos
frame_counter = 0  # Contador de fotogramas para estabilizar la letra detectada
stabilization_frames = 13  # Número de fotogramas consecutivos para estabilizar una letra
words_history = []  # Lista para almacenar las últimas palabras

# Configuración para la cámara virtual (resolución 640x480, 30 FPS)
# No es necesario para este entorno de servidor web

@app.route('/')
def index():
    # Página principal (si necesitas una interfaz web básica)
    return render_template('index.html')  # Asegúrate de tener este archivo si usas plantillas

@app.route('/predict', methods=['POST'])
def predict():
    # Procesar la imagen enviada por el cliente
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decodificar la imagen en base64
    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convertir la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(image_rgb)

    class_label = ""  # Predicción de la seña
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer las coordenadas x, y, z de cada punto clave
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Convertir a numpy y redimensionar para el modelo
            keypoints = np.array(keypoints).reshape(1, -1)

            # Realizar la predicción sin mostrar la barra de progreso
            try:
                prediction = model.predict(keypoints, verbose=0)  # verbose=0 para desactivar la barra de progreso
                class_index = np.argmax(prediction)
                class_label = class_names[class_index]  # Obtener la clase predicha
            except IndexError as e:
                class_label = "Etiqueta no encontrada"  # En caso de que la predicción no esté en el índice
                print(f"Error de índice: {e}. Usando 'Etiqueta no encontrada'.")

            # Manejar la palabra y estabilización
            if class_label == last_letter:
                frame_counter += 1
            else:
                frame_counter = 0  # Reiniciar el contador si cambia la letra
                last_letter = class_label  # Actualizar la última letra

            if frame_counter >= stabilization_frames:
                current_word += class_label
                frame_counter = 0  # Reiniciar el contador después de añadir la letra

    # Devolver la predicción
    return jsonify({"predicted_class": class_label, "current_word": current_word})

if __name__ == '__main__':
    # Ejecutar la app de Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
