import os
import sys
import time
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, Response
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
app = Flask(_name_)

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

# Variables para manejar el tiempo
last_detection_time = time.time()  # Último momento en que se detectó una letra
timeout = 2  # Tiempo en segundos sin detección para borrar la palabra
clear_all_timeout = 5  # Tiempo para borrar todas las palabras después de 5 segundos sin detección
word_complete_timeout = 1  # Tiempo sin detección para considerar que la palabra está completa

# Ruta principal de la aplicación web
@app.route('/')
def home():
    return render_template("index.html")  # Asegúrate de tener un archivo HTML para renderizar la página

# Función para capturar el video y procesarlo
def generate_frames():
    global current_word, last_letter, words_history, last_detection_time  # Hacer que las variables sean globales

    cap = cv2.VideoCapture(0)  # Captura de la cámara local

    if not cap.isOpened():
        return jsonify({"error": "No se pudo acceder a la cámara."}), 500

    while True:
        success, image = cap.read()
        if not success:
            break

        # Convertir la imagen a RGB
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

                # Si la letra es igual a la última detectada, incrementar el contador
                if class_label == last_letter:
                    frame_counter += 1
                else:
                    frame_counter = 0  # Reiniciar el contador si cambia la letra
                    last_letter = class_label  # Actualizar la última letra

                # Si se estabiliza por los fotogramas requeridos, añadir la letra
                if frame_counter >= stabilization_frames:
                    current_word += class_label
                    frame_counter = 0  # Reiniciar el contador después de añadir la letra

                # Actualizar el tiempo de la última detección
                last_detection_time = time.time()

        # Borrar la palabra si no hay detección durante el tiempo especificado
        if time.time() - last_detection_time > timeout:
            if current_word:
                words_history.append(current_word)
                # Si tenemos 4 palabras, eliminamos las 3 primeras
                if len(words_history) > 3:
                    words_history = words_history[1:]  # Elimina las 3 primeras palabras
            current_word = ""  # Limpiar la palabra

        # Si han pasado más de 5 segundos sin detección de ninguna seña, borrar todas las palabras
        if time.time() - last_detection_time > clear_all_timeout:
            words_history = []  # Borrar todas las palabras

        # Si pasa más de 'word_complete_timeout' sin detectar ninguna letra, considerar la palabra como completa
        if time.time() - last_detection_time > word_complete_timeout and current_word:
            words_history.append(current_word)
            if len(words_history) > 3:
                words_history = words_history[1:]  # Elimina las 3 primeras palabras
            current_word = ""  # Limpiar la palabra

        # Mostrar las palabras formadas, en fila horizontal
        text_word = " ".join(words_history) + current_word
        image_height, image_width, _ = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        # Mostrar la palabra formada
        (text_width, text_height), _ = cv2.getTextSize(text_word, font, font_scale, font_thickness)
        text_x = (image_width - text_width) // 2  # Posición centrada en la parte inferior
        text_y = image_height - 50  # Parte inferior

        # Añadir contorno y sombra al texto de la palabra
        cv2.putText(image, text_word, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(image, text_word, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

        # Convertir la imagen a formato JPEG para enviarla como un frame
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Enviar el frame como un flujo de bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Ruta para mostrar el video en el navegador (streaming)
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Iniciar el servidor Flask
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)
