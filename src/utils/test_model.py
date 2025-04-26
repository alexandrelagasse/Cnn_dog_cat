import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def load_image(img_path, target_size=(64, 64)):
    """Charge et prétraite une image pour le modèle"""
    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x = x / 255.0  # Normalisation
    return x, img

def predict_image(model, image_path):
    """Prédit la classe d'une image"""
    x, original_img = load_image(image_path)
    prediction = model.predict(x)
    return prediction[0][0], original_img

def display_prediction(image, prediction, title):
    """Affiche l'image avec sa prédiction"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"{title}\n{'Chien' if prediction > 0.5 else 'Chat'} ({prediction:.2f})")
    plt.axis('off')
    plt.show()

def main():
    # Chemin vers le modèle
    model_path = 'modele/cnn_chien_chat_ameliore.h5'
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle n'a pas été trouvé à l'emplacement : {model_path}")
        return
    
    # Charger le modèle avec des options de compatibilité
    try:
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            options=tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost'
            )
        )
        print("Modèle chargé avec succès!")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {str(e)}")
        return
    
    # Images spécifiques à tester
    test_images = [
        'tester/cjpeg',
        'tester/OIP.jpeg',
        'tester/R.jpeg'
    ]
    
    print(f"\nTest des {len(test_images)} images :")
    
    # Tester chaque image
    for img_path in test_images:
        try:
            prediction, original_img = predict_image(model, img_path)
            display_prediction(original_img, prediction, f"Image: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Erreur lors du traitement de {img_path}: {str(e)}")

if __name__ == "__main__":
    main() 