# Se importan las librerías necesarias
import cv2
import pickle
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Se carga el modelo
model = load_model("model/model_emotion_recognition.h5")

# Se carga el diccionario de las emociones utilizadas en el modelo
with open("model/model_emotions_dictionary.pickle", "rb") as text_file:
    emotions_dictionary = pickle.load(text_file)

# A continuación, el código para la detección de rostros utilizando
# Mediapipe
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Utilizar la webcam
cap = cv2.VideoCapture(0)

# Utilizar un video
#cap = cv2.VideoCapture('test-resources/video_test.mp4')

# Utilizar una imagen
#cap = cv2.VideoCapture('test-resources/image_test.jpg')

with mp.solutions.face_mesh.FaceMesh(
    # Máximo número de rostros que se pueden detectar a la vez
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Si se detecta un rostro
        if results.multi_face_landmarks is not None:
            # Por cada rostro detectado
            for face_landmarks in results.multi_face_landmarks:
                # Se obtienen los puntos de la Face Mesh
                landmarks = np.array(
                    [(lmk.x, lmk.y) for lmk in face_landmarks.landmark]
                )
                # Se obtienen los puntos que indican:
                #  - El ancho de la Face Mesh (de x_min hasta x_max)
                #  - La altura de la Face Mesh (de y_min hasta y_max)
                x_min = int(landmarks[:, 0].min() * image.shape[1])
                y_min = int(landmarks[:, 1].min() * image.shape[0])
                x_max = int(landmarks[:, 0].max() * image.shape[1])
                y_max = int(landmarks[:, 1].max() * image.shape[0])
                # Se recorta el rostro detectado empleando las esquinas de la Face Mesh
                face = image[
                    y_min - 15:y_max + 15, x_min - 15: x_max + ((y_max - y_min) - (x_max - x_min)) + 15
                ]
                # Si el tamaño del rostro es mayor que cero, es decir, existe un rostro
                if face.shape[0] > 0 and face.shape[1] > 0:
                    # Se cambia el tamaño del rostro al tamaño que acepta el modelo
                    face = cv2.resize(face, (48, 48))
                    # Se cambia el rostro a escala de grises
                    im_rgb = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                    # Se hace la predicción de la emoción del rostro detectado utilizando el modelo
                    emotion_prediction = model.predict(np.expand_dims(np.expand_dims(im_rgb, -1), 0))
                    # Se obtiene la emoción con el índice más alto
                    maxindex = int(np.argmax(emotion_prediction))
                    # Se muestra la emoción predicha encima del rostro
                    cv2.putText(
                        image,
                        emotions_dictionary[maxindex],
                        (x_min, y_min),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 204),
                        2,
                        cv2.LINE_AA,
                    )
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                )
        cv2.imshow("MediaPipe Face Mesh", image)
        # Pulsar tecla 'esc' para salir
        if cv2.waitKey(5) & 0xFF == 27:
            cap.release()
            break
    # Pulsar tecla 'esc' para salir
    while cap.isOpened():
        if cv2.waitKey(5) & 0xFF == 27:
            break
    print("\n FUENTE DESACTIVADA \n")

cap.release()

