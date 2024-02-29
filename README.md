# Emotion Recognition Model

## Descripción
Este proyecto crea un modelo de reconocimiento de emociones que se puede
utilizar en tiempo real mediante una webcam, video o imagen.
El proyecto está implementado en Python y utiliza varias bibliotecas y
herramientas de inteligencia artificial y procesamiento de datos e imágenes.

***Es importante mencionar que todos los archivos del proyecto se pueden ejecutar en una máquina local. Sin embargo, se recomienda utilizar el entorno en Kaggle que se ha preparado para el proceso de entrenamiento del modelo, ya que en una máquina local se consumen muchos recursos y puede tardar demasiado. En los apartados 'Instalación y requisitos previos' y 'Uso del proyecto' se detallan los pasos que hay que seguir en ambos casos.**

***Enlace a entorno de Kaggle: [Emotion-Recognition-Model](https://www.kaggle.com/code/eduuoc/emotion-recognition-model)**

## Estructura del proyecto
El proyecto está estructurado de la siguiente manera:

- Carpeta **'dataset'**. Una vez ejecutado el archivo 'data_preparation.py',
contiene las imagenes que utiliza el modelo clasificadas según su emocion.

- Carpeta **'FER-2013'**. Contiene archivos del dataset FER-2013, en concreto, se
utiliza el archivo 'fer2013.csv' que almacena los datos que emplea el modelo.

- Carpeta **'model'**. Una vez ejecutado el archivo 'model_training.py',
contiene el modelo entrenado e información sobre él.

- Carpeta **'stats'**. Una vez ejecutado el archivo 'model_stats.py', contiene
imágenes que muestran estadísticas del modelo entrenado.

- Carpeta **'test-resources'**. Contiene una imagen y un video de ejemplo para
que se pueda probar el modelo en ellos.

- Archivo **'data_preparation.py'**. Se utiliza para obtener y preparar los datos
del dataset FER-2013.

- Documento **'LICENSE.txt'**. Contiene información relacionada con la licencia
de este proyecto.

- Archivo **'model_stats.py'**. Se utiliza para obtener estadísticas del modelo
entrenado.

- Archivo **'model_test.py'**. Se utiliza para probar el modelo mediante webcam,
video o imagen.

- Archivo **'model_training.py'**. Se utiliza para entrenar el modelo.

- Documento **'README.md'**. Contiene información sobre el proyecto, como el
proceso de instalación o su uso.

- Documento **'requirements.txt'**. Contiene todas las dependencias del proyecto.

## Instalación y requisitos previos

### Máquina local

Para utilizar este proyecto es necesario tener instaladas las siguientes librerías:

```bash
pip install pandas
pip install mediapipe
pip install tensorflow
pip install -U scikit-learn
```
También se puede instalar todas las dependencias del proyecto utilizando el archivo 'requirements.txt':

```bash
pip install -r requirements.txt
```

### Kaggle

Para ejecutar el código de Kaggle lo ideal es primero asignarle una GPU para que vaya mucho más rápido. 

## Uso del proyecto

En este apartado se muestran los pasos que hay que seguir para utilizar el proyecto tanto en un máquina local como en Kaggle, pero antes, hay que tener en cuenta que:

**- El modelo ya viene entrenado, por lo que todos los pasos de 'Kaggle' y los pasos 1, 2, 3 y 4 de 'Máquina local' son innecesarios si solo se quiere probar el modelo, para lo que solo es necesario realizar el paso 5 de 'Máquina local'. Si por lo contrario, se quiere modificar algo del modelo, dado que el modelo habrá cambiado, será necesario realizar todos los pasos para volver a entrenar el modelo.**

**- El proyecto se puede ejecutar en Kaggle para acelerar el proceso de entrenamiento del modelo, sin embargo, no se puede probar el modelo de reconocimiento de emociones en tiempo real en dicho sito, para hacerlo, una vez entrenado el modelo en Kaggle, se tiene que descargar la carpeta 'model' y reemplazarla por la existente en una máquina local, donde si que se podrá probar el modelo entrenado.**

### Máquina local

Pasos ha seguir:

1. Instalar las librerías mencionadas en la sección 'Instalación y requisitos previos'.
2. Ejecutar el archivo 'data_preparation.py' que creará la carpeta 'dataset' con las imagenes utilizadas para entrenar el modelo.
3. Ejecutar el archivo 'model_training.py' para entrenar el modelo y que también creará la carpeta 'model' donde se almacenará el modelo entrenado.
4. Ejecutar el archivo 'model_stats.py' que creará la carpeta 'stats' con imágenes que muestran estadísticas del modelo entrenado.
5. *(Hacer en máquina local, no en Kaggle)* Ejecutar el archivo 'model_test.py' para probar el modelo. En este archivo se encuentran las siguientes lineas:

    ```python
    # Utilizar la webcam
    cap = cv2.VideoCapture(0)

    # Utilizar un video
    #cap = cv2.VideoCapture('test-resources/video_test.mp4')

    # Utilizar una imagen
    #cap = cv2.VideoCapture('test-resources/image_test.jpg')
    ```
    Descomentar solo una de las tres lineas de código según se quiera utilizar la webcam, un video o una imagen. Por defecto, viene descomentada la línea para utilizar la webcam.
    Además, en la carpeta 'test-resources' hay un video y una imagen para probar el modelo.

### Kaggle

Pasos ha seguir:

1. Acceder al código en [Kaggle](https://www.kaggle.com/code/eduuoc/emotion-recognition-model).
2. Seguir los pasos indicados en el entorno de trabajo de Kaggle.
3. Pasar al paso 5 de 'Máquina local' para probar el modelo.

## Crédito
Este proyecto utiliza:

- El conjunto de datos FER-2013: [FER-2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

- MediaPipe Face Mesh para Python: [MediaPipe Face Mesh](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md)

- Diversos recursos de Internet para realizar el diseño del modelo, la preparación de datos y la extracción de estadísticas.

- Vídeo de ejemplo: [video_test](https://www.youtube.com/watch?v=ocPjB4szjM0&t=91s&ab_channel=ImagineVideoclips)

- Imagen de ejemplo: [image_test](https://qph.cf2.quoracdn.net/main-qimg-ed1dcf6e8956499bd0e7571aaa8b44fa-lq)

## Licencia
El código de este proyecto está bajo la licencia [MIT](https://opensource.org/license/mit/). Consultar el archivo 'LICENSE.txt' para obtener más detalles.