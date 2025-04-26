import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image for the model"""
    try:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x = x / 255.0  # Normalization
        return x
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {img_path}: {str(e)}")
        return None

def predict_image(model, image_path):
    """Predict the class of an image"""
    x = load_image(image_path)
    if x is None:
        return None
    try:
        prediction = model.predict(x)
        return prediction[0][0]
    except Exception as e:
        print(f"Erreur lors de la prédiction pour {image_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test the CNN model on images')
    parser.add_argument('--model', type=str, default='src/model/cnn_dog_cat_improved.h5',
                      help='Path to the model file')
    parser.add_argument('--image', type=str, help='Path to a single image to test')
    parser.add_argument('--test_dir', type=str, default='test_images',
                      help='Directory containing test images')
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Erreur: Modèle non trouvé à: {args.model}")
        return

    # Load model
    try:
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(args.model)
        print("Modèle chargé avec succès!")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return

    # Test single image or directory
    if args.image:
        try:
            prediction = predict_image(model, args.image)
            if prediction is not None:
                result = "Chien" if prediction > 0.5 else "Chat"
                confidence = max(prediction, 1 - prediction)  # Prendre la plus grande valeur entre prediction et 1-prediction
                print(f"\nImage: {os.path.basename(args.image)}")
                print(f"Résultat: {result} (confiance: {confidence:.2%})")
        except Exception as e:
            print(f"Erreur lors du traitement de {args.image}: {str(e)}")
    else:
        # Test all images in directory
        if not os.path.exists(args.test_dir):
            print(f"Erreur: Dossier de test non trouvé: {args.test_dir}")
            return

        test_images = [os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nTest de {len(test_images)} images:")
        
        for img_path in test_images:
            try:
                prediction = predict_image(model, img_path)
                if prediction is not None:
                    result = "Chien" if prediction > 0.5 else "Chat"
                    confidence = max(prediction, 1 - prediction)  # Prendre la plus grande valeur entre prediction et 1-prediction
                    print(f"\nImage: {os.path.basename(img_path)}")
                    print(f"Résultat: {result} (confiance: {confidence:.2%})")
            except Exception as e:
                print(f"Erreur lors du traitement de {img_path}: {str(e)}")

if __name__ == "__main__":
    main() 