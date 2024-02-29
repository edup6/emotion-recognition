# Se importan las librerías necesarias
import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.utils import load_img, img_to_array
from matplotlib.ticker import FixedLocator
from keras.preprocessing.image import ImageDataGenerator

# Si no existe la carpeta 'stats' se crea
if not os.path.exists("stats"):
    os.makedirs("stats")

# Se carga el modelo
model = load_model("model/model_emotion_recognition.h5")

# Se carga el historial de entrenamiento del modelo
history = pd.read_csv('model/model_history.csv', sep=',')

# Se guarda el gráfico Accuracy/Epoch
fig1 = plt.figure()
plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.title(
    "Accuracy/Epoch \n\n"
    + "Final val_accuracy: "
    + str(round(history["val_accuracy"].iloc[-1] * 100, 2))
    + "%"
)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
current_values = plt.gca().get_yticks()
plt.gca().yaxis.set_major_locator(FixedLocator(current_values))
plt.gca().set_yticklabels(["{:.2f}".format(i * 100) for i in plt.gca().get_yticks()])
plt.tight_layout()
plt.savefig("stats/accuracy_epoch.png")

# Se guarda el gráfico Loss/Epoch
fig2 = plt.figure()
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title(
    "Loss/Epoch \n\n"
    + "Final val_loss: "
    + str(round(history["val_loss"].iloc[-1], 2))
)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.tight_layout()
plt.savefig("stats/loss_epoch.png")

# Se obtienen las imágenes de validación usadas en el modelo (poner
# mismos parámetros que en 'model_train.py')
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)
val_generator = val_datagen.flow_from_directory(
    "dataset",
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
)
val_generator.reset()

# Se guarda la matriz de confusión
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
confusion_matrix = metrics.confusion_matrix(y_true, y_pred_classes)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=val_generator.class_indices
)
cm_display.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("stats/confusion_matrix.png")

# Se guarda el gráfico que muestra la precisión del modelo para cada
# emoción
total_predictions = np.sum(confusion_matrix, axis=1)
correct_predictions = np.diag(confusion_matrix)
accuracy_percentages = correct_predictions / total_predictions * 100
emotion_names = [key.capitalize() for key in val_generator.class_indices.keys()]
fig, ax = plt.subplots()
ax.bar(emotion_names, accuracy_percentages)
ax.set_ylim([0, 100])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Emotion")
ax.set_title("Accuracy/Emotion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("stats/accuracy_emotion.png")

# Obtener índices de las capas convolucionales del modelo
conv_layer_index = []
for i in range(len(model.layers)):
    if "conv2d" in model.layers[i].name:
        conv_layer_index.append(i)

# Se guardan los filtros de cada capa convolucional de la CNN
for layer_index in conv_layer_index:
    layer = model.layers
    filters, biases = model.layers[layer_index].get_weights()
    fig1 = plt.figure(figsize=(8, 12))
    columns = 8
    rows = 4
    n_filters = columns * rows
    for i in range(1, n_filters + 1):
        f = filters[:, :, :, i - 1]
        fig1 = plt.subplot(rows, columns, i)
        fig1.set_xticks([])
        fig1.set_yticks([])
        plt.imshow(f[:, :, 0], cmap="gray")
    plt.tight_layout()
    plt.savefig("stats/filters_conv_layer_" + str(layer_index) + ".png")

# Se guardan las características extraídas en cada capa convolucional
# de la CNN
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
# Seleccionar imagen del conjunto de datos para ver sus características
img = load_img("dataset/angry/0.png", color_mode="grayscale", target_size=(48, 48))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
feature_output = model_short.predict(img)
columns = 8
rows = 4
iteration = 0
for ftr in feature_output:
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, columns * rows + 1):
        fig = plt.subplot(rows, columns, i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i - 1], cmap="gray")
    plt.tight_layout()
    plt.savefig(
        "stats/features_conv_layer_" + str(conv_layer_index[iteration]) + ".png"
    )
    iteration += 1