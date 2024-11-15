import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Cambia la ruta al modelo según corresponda
modelo = load_model('/Users/santiagovelascocobo/Documents/redneuronal2/modelo_gatos_perros_mobilenet.keras')

# Función para clasificar imágenes
def clasificar_imagen(imagen_path):
    imagen = load_img(imagen_path, target_size=(128, 128))  # Ajusta el tamaño según tu modelo
    imagen_array = img_to_array(imagen) / 255.0  # Normaliza la imagen
    imagen_array = imagen_array.reshape(1, 128, 128, 3)  # Cambia la forma de la imagen
    prediccion = modelo.predict(imagen_array)
    return 'Gato' if prediccion[0][0] > 0.5 else 'Perro'  # Ajusta según la salida de tu modelo


# Clasificar imágenes en la carpeta 'images/gatos'
carpeta_gatos = '/Users/santiagovelascocobo/Documents/redneuronal2/images/gatos'
for imagen in os.listdir(carpeta_gatos):
    if imagen.endswith('.jpg') or imagen.endswith('.png'):
        resultado = clasificar_imagen(os.path.join(carpeta_gatos, imagen))
        print(f"{imagen} es: {resultado}")

# Pausa para que puedas analizar los resultados de los gatos
input("Presiona Enter para continuar con la clasificación de perros...")

# Clasificar imágenes en la carpeta 'images/perros'
carpeta_perros = '/Users/santiagovelascocobo/Documents/redneuronal2/images/perros'
for imagen in os.listdir(carpeta_perros):
    if imagen.endswith('.jpg') or imagen.endswith('.png'):
        resultado = clasificar_imagen(os.path.join(carpeta_perros, imagen))
        print(f"{imagen} es: {resultado}")

