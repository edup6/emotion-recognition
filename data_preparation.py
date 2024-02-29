# Se importan las librerías necesarias
import os
import pandas as pd
from PIL import Image
import shutil

# Se carga el archivo CSV que contiene el conjunto de datos FER-2013
data = pd.read_csv("FER-2013/fer2013/fer2013.csv")

# Se crea un diccionario para asociar los índices de las emociones del
# CSV con el nombre de su emoción
emotion_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# Se crea la carpeta 'dataset' si todavía no está creada
if not os.path.exists("dataset"):
    os.makedirs("dataset")
# Se elimina la carpeta 'dataset' si ya está esta creada y se vuelve a crear
else:
    shutil.rmtree('dataset')
    os.makedirs("dataset")

# Se recorren las filas del CSV y en cada una se:
#  - obtiene el índice de la emoción
#  - cambia el string de píxeles a una lista de strings
#  - cada string de la lista se cambia a un número entero
#  - se crea una nueva imagen
#  - se añaden los píxeles a la nueva imagen
#  - si todavía no existe la carpeta con el nombre de la emoción leída,
#    se crea
#  - se guarda la imagen en su respectiva carpeta
for index, row in data.iterrows():
    emotion = row["emotion"]
    pixels = row["pixels"].split()
    pixels = [int(pixel) for pixel in pixels]
    img = Image.new("L", (48, 48))
    img.putdata(pixels)

    if not os.path.exists(f"dataset/{emotion_dict[emotion]}"):
        os.makedirs(f"dataset/{emotion_dict[emotion]}")

    img.save(f"dataset/{emotion_dict[emotion]}/{index}.png")

#Si se quiere eliminar alguna emoción para el modelo
#for index, row in data.iterrows():
#    emotion = row["emotion"]
#    if emotion_dict[emotion] not in ['disgust', 'fear', 'sad']:
#        pixels = row["pixels"].split()
#        pixels = [int(pixel) for pixel in pixels]
#        img = Image.new("L", (48, 48))
#        img.putdata(pixels)
#
#        if not os.path.exists(f"dataset/{emotion_dict[emotion]}"):
#            os.makedirs(f"dataset/{emotion_dict[emotion]}")
#
#        img.save(f"dataset/{emotion_dict[emotion]}/{index}.png")