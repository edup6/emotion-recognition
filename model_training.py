# Se importan las librerías necesarias
import os
import time
import pickle
import shutil
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Si no existe la carpeta 'model' se crea
if not os.path.exists("model"):
    os.makedirs("model")
# Se elimina la carpeta 'model' si ya está esta creada y se vuelve a crear
else:
    shutil.rmtree('model')
    os.makedirs("model") 

# Se establecen los parámetros de la CNN
batch_size = 128
epochs = 100
img_size = (48, 48)
validation_split = 0.2
train_dir = "dataset"

# Se cargan y procesan los datos de entrenamiento y validación
train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255, 
    validation_split = validation_split,
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=10
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = img_size,
    batch_size = batch_size,
    color_mode = "grayscale",
    class_mode = "categorical",
    subset = "training",
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = img_size,
    batch_size = batch_size,
    color_mode = "grayscale",
    class_mode = "categorical",
    subset = "validation",
)

# Se establece la arquitectura de la CNN
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',  kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',  kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Se configuran los callbacks
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 7, verbose = 1)
model_checkpoint = ModelCheckpoint("model/model_emotion_recognition.h5", save_best_only = True, monitor = 'val_accuracy', mode = 'max', verbose = 1)
history_logger = tf.keras.callbacks.CSVLogger('model/model_history.csv', separator=",", append=True)

# Se configura el learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.0001,
    decay_steps = 5000,
    decay_rate = 0.96,
    staircase = True
)

# Se compila el modelo
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), metrics = ['accuracy'])

# Se inicia un contador para saber cuanto tarda el entrenamiento
# del modelo
start_time = time.time()

# Se comienza a entrenar el modelo
history = model.fit(
    train_generator,
    epochs = epochs,
    validation_data = val_generator,
    verbose = 1,
    callbacks = [early_stopping, model_checkpoint, LearningRateScheduler(lr_schedule), history_logger]
)

# Se pausa el contador, ha terminado el entrenamiento
end_time = time.time()

# Variable que almacena la duración del entrenamiento
training_time = end_time - start_time

# Se reinicia el generador de datos de validación para que sean los
# mismos datos
val_generator.reset()   

# Se guarda el tiempo de entrenamiento del modelo en un archivo de texto
with open("model/model_training_duration.txt", "w") as text_file:
    text_file.write(
        "Tiempo de entrenamiento: "
        + time.strftime("%Hh:%Mm:%Ss", time.gmtime(training_time))
    )
    
# Se crea un diccionario con las emociones utilizadas en este modelo
emotions_dictionary = {v: k for k, v in train_generator.class_indices.items()}

# Se guarda el diccionario en un archivo
with open("model/model_emotions_dictionary.pickle", "wb") as text_file:
    pickle.dump(emotions_dictionary, text_file)