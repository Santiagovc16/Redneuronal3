import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import hashlib
import cv2  # Para la cámara

# Configuración de las rutas
base_dir = 'images'  # Directorio base para entrenamiento
venom_dir = 'venom'  # Directorio donde están las imágenes de prueba
que_es_dir = 'que_es'  # Directorio donde están esgato.jpg y esperro.jpg

# Verificar que los directorios existen
for dir in [base_dir, venom_dir, que_es_dir]:
    if not os.path.exists(dir):
        print(f"El directorio {dir} no existe.")
        exit()  # Salir si algún directorio no existe

# Cargar modelo existente si ya fue entrenado
model_path = 'modelo_gatos_perros_mobilenet.keras'
needs_training = not os.path.exists(model_path)

# Crear un hash del modelo y configuraciones de entrenamiento
def get_config_hash():
    config = '''
    base_dir: {}
    datagen_params: rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    model_structure: MobileNetV2 with Dense(128, relu) + Dropout(0.5) + Dense(2, softmax)
    '''.format(base_dir)
    return hashlib.md5(config.encode()).hexdigest()

# Comprobar si el archivo de hash de configuración existe y coincide
hash_file = 'config_hash.txt'
current_hash = get_config_hash()

if os.path.exists(hash_file):
    with open(hash_file, 'r') as file:
        saved_hash = file.read().strip()
    if saved_hash != current_hash:
        needs_training = True
else:
    needs_training = True

if needs_training:
    # Configuración del aumento de datos
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generador de datos
    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    # Cargar el modelo MobileNetV2 preentrenado
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Congelar las capas de la base
    for layer in base_model.layers:
        layer.trainable = False

    # Añadir nuevas capas de clasificación
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Añadir Dropout para reducir el sobreajuste
    output = layers.Dense(2, activation='softmax')(x)

    # Definir el nuevo modelo
    model = models.Model(inputs=base_model.input, outputs=output)

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(train_generator, epochs=30)  # Sin paralelización en la carga de datos

    # Guardar el modelo
    model.save(model_path)

    # Guardar el hash actual en el archivo
    with open(hash_file, 'w') as file:
        file.write(current_hash)
else:
    # Cargar el modelo ya entrenado
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado desde el archivo guardado.")

# Función para mostrar la confianza de las predicciones en una gráfica de barras
def mostrar_predicciones(prediccion, clases, clase_correcta):
    plt.bar(clases, prediccion[0], color=['red' if c != clase_correcta else 'green' for c in clases])
    plt.xlabel('Clases')
    plt.ylabel('Confianza')
    plt.title('Confianza de la Clasificación')
    plt.show()

# Función para cargar y predecir, con visualización de confianza
def cargar_y_predecir(imagen_path):
    try:
        # Cargar la imagen
        img = image.load_img(imagen_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar la predicción
        prediccion = model.predict(img_array)
        clase_predicha = np.argmax(prediccion)
        clases = ['Gato', 'Perro']

        # Determinar la clase y la imagen a mostrar
        if clase_predicha == 0:  
            clase = "Gato"
            imagen_mostrar = os.path.join(que_es_dir, 'esgato.jpg')
            clase_correcta = "Gato"  
        else:  
            clase = "Perro"
            imagen_mostrar = os.path.join(que_es_dir, 'esperro.jpg')
            clase_correcta = "Perro" 

        # Cargar y mostrar la imagen correspondiente
        img_mostrar = image.load_img(imagen_mostrar, target_size=(128, 128))
        plt.imshow(img_mostrar)
        plt.axis('off')  # No mostrar ejes
        plt.title(f"{os.path.basename(imagen_path)} - Identificado como: {clase}")
        plt.show()  # Mostrar la imagen

        # Verificar si el modelo se equivocó
        if (clase_predicha == 0 and clase_correcta == "Perro") or (clase_predicha == 1 and clase_correcta == "Gato"):
            print(f"¡Error! Se identificó como {clase}, pero debería ser {clase_correcta}.")

        # Mostrar la gráfica de confianza
        mostrar_predicciones(prediccion, clases, clase_correcta)

        return clase
    except Exception as e:
        print(f"Error al procesar {imagen_path}: {e}")
        return "Error"

# Función para activar la cámara, capturar y analizar imágenes de la carpeta 'venom'
def activar_camara():
    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)

    while True:
        # Leer el cuadro de la cámara
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar la imagen.")
            break

        # Mostrar el cuadro
        cv2.imshow('Presiona "q" para salir', frame)

        # Esperar a que se presione la tecla 's' para capturar la imagen
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Guardar la imagen capturada
            imagen_capturada_path = 'venom/captura.jpg'
            cv2.imwrite(imagen_capturada_path, frame)
            print(f"Imagen guardada como {imagen_capturada_path}")

            # Realizar la predicción en la imagen capturada
            resultado = cargar_y_predecir(imagen_capturada_path)
            print(f"La imagen capturada es: {resultado}")

            # Analizar imágenes adicionales en la carpeta 'venom'
            print("\nAnalizando imágenes adicionales en la carpeta 'venom'...")
            for filename in os.listdir(venom_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(venom_dir, filename)
                    resultado_venom = cargar_y_predecir(image_path)
                    print(f"{filename} es: {resultado_venom}")

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para activar la cámara
activar_camara()
