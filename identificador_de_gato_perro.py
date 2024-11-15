import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import json
from datetime import datetime

# Configuración de las rutas
base_dir = 'images'
venom_dir = 'venom'
que_es_dir = 'que_es'
train_info_path = 'train_info.json'  # Archivo para guardar la información de entrenamiento

# Verificar que los directorios existen
for dir in [base_dir, venom_dir, que_es_dir]:
    if not os.path.exists(dir):
        print(f"El directorio {dir} no existe.")
        exit()

# Contar el número de imágenes en el directorio de entrenamiento
def contar_imagenes(directorio):
    return sum([len(files) for _, _, files in os.walk(directorio)])

# Cargar información previa de entrenamiento si existe
if os.path.exists(train_info_path):
    with open(train_info_path, 'r') as f:
        train_info = json.load(f)
else:
    train_info = {}

# Verificar si es necesario entrenar
num_imagenes_actual = contar_imagenes(base_dir)
modelo_existente = os.path.exists('modelo_gatos_perros_y_otros.keras')
entrenar = True

# Comprobar si se han agregado nuevas imágenes o no se ha entrenado previamente
if modelo_existente and train_info.get('num_imagenes') == num_imagenes_actual:
    print("Modelo ya entrenado recientemente. Solo se realizará la verificación.")
    entrenar = False
else:
    print("Nuevo entrenamiento requerido.")
    entrenar = True

# Configuración del dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,  # 80% entrenamiento, 20% validación
    subset='training',
    seed=123,
    label_mode='categorical',
    image_size=(128, 128),
    batch_size=128
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    label_mode='categorical',
    image_size=(128, 128),
    batch_size=128
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Cargar o entrenar el modelo
if entrenar:
    # Cargar el modelo MobileNetV2 preentrenado
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Descongelar las últimas 20 capas del modelo base
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Añadir nuevas capas de clasificación
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(3, activation='softmax')(x)  # Cambiado a 3 clases

    model = models.Model(inputs=base_model.input, outputs=output)

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenar el modelo (20 épocas)
    with tf.device('/GPU:0'):
        history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[reduce_lr, early_stopping])

    # Guardar el modelo y la información de entrenamiento
    model.save('modelo_gatos_perros_y_otros.keras')
    train_info = {
        'num_imagenes': num_imagenes_actual,
        'last_trained': datetime.now().isoformat()
    }
    with open(train_info_path, 'w') as f:
        json.dump(train_info, f)

    # Graficar precisión y pérdida
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.show()

else:
    # Cargar el modelo existente
    model = models.load_model('modelo_gatos_perros_y_otros.keras')

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
        img = image.load_img(imagen_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediccion = model.predict(img_array)
        clase_predicha = np.argmax(prediccion)
        clases = ['Gato', 'Perro', 'No_perro_gato']  # Incluye la nueva clase

        if clase_predicha == 0:  
            clase = "Gato"
        elif clase_predicha == 1:  
            clase = "Perro"
        else:  
            clase = "No_perro_gato"

        mostrar_predicciones(prediccion, clases, clase)
        return clase
    except Exception as e:
        print(f"Error al procesar {imagen_path}: {e}")
        return "Error"

# Analizar todas las imágenes en la carpeta 'venom'
for filename in os.listdir(venom_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        animal_path = os.path.join(venom_dir, filename)
        resultado = cargar_y_predecir(animal_path)
        print(f"{filename} es: {resultado}")
