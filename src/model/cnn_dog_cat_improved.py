import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
import multiprocessing

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Vérification de l'installation TensorFlow
    logging.info("Vérification de l'installation TensorFlow...")
    logging.info(f"Version TensorFlow: {tf.__version__}")
    logging.info(f"Version CUDA disponible: {tf.sysconfig.get_build_info()['cuda_version']}")
    logging.info(f"Version cuDNN disponible: {tf.sysconfig.get_build_info()['cudnn_version']}")

    # Configuration du CPU et GPU
    logging.info("Configuration du CPU et GPU...")
    NUM_WORKERS = min(8, multiprocessing.cpu_count() - 1)
    logging.info(f"Nombre de workers CPU disponibles: {NUM_WORKERS}")

    # Configuration des paramètres d'entraînement
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 24  # Réduit de 32 à 24
    EPOCHS = 50

    logging.info(f"Paramètres d'entraînement:")
    logging.info(f"- Taille des images: {IMG_SIZE}")
    logging.info(f"- Taille du batch: {BATCH_SIZE}")
    logging.info(f"- Nombre d'époques: {EPOCHS}")

    # Configuration GPU pour RTX 4060
    logging.info("Configuration du GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)]  # Augmenté à 8GB
            )
            logging.info(f"GPU configuré avec succès: {gpus[0]}")
        except RuntimeError as e:
            logging.error(f"Erreur de configuration GPU: {e}")
    else:
        logging.warning("Aucun GPU détecté. Vérifiez l'installation de CUDA et cuDNN.")

    # Configuration de la précision avec mixed_float16
    logging.info("Configuration de la précision...")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    logging.info("Précision configurée: mixed_float16")

    # Configuration des chemins
    TRAIN_DIR = 'data/training_set/training_set'
    VAL_DIR = 'data/test_set/test_set'
    MODEL_PATH = 'modele/cnn_chien_chat_ameliore.h5'
    LOG_DIR = 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Augmentation des données avancée
    logging.info("Configuration de l'augmentation des données...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        channel_shift_range=20
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Création d'une classe personnalisée pour le générateur de données
    class CustomDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, generator, **kwargs):
            super().__init__(**kwargs)
            self.generator = generator
            self.classes = generator.classes  # Accès aux classes du générateur original
            
        def __len__(self):
            return len(self.generator)
            
        def __getitem__(self, idx):
            return self.generator[idx]
            
        def on_epoch_end(self):
            self.generator.on_epoch_end()

    # Création des générateurs
    logging.info("Création des générateurs de données...")
    train_generator = CustomDataGenerator(
        train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True
        )
    )

    validation_generator = CustomDataGenerator(
        val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
    )

    logging.info(f"Données d'entraînement: {len(train_generator)} batches")
    logging.info(f"Données de validation: {len(validation_generator)} batches")

    # Calcul des poids de classe
    logging.info("Calcul des poids de classe...")
    class_counts = np.bincount(train_generator.classes)
    total_samples = sum(class_counts)
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    logging.info(f"Poids de classe calculés: {class_weights}")

    # Construction du modèle optimisé
    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),  # Ajout d'une couche Input explicite
        # Premier bloc de convolution
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Deuxième bloc de convolution
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Troisième bloc de convolution
        Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Quatrième bloc de convolution
        Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Cinquième bloc de convolution
        Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Couches denses
        Flatten(),
        Dense(2048, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compilation du modèle avec paramètres optimisés
    initial_learning_rate = 0.001
    optimizer = Adam(learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Callbacks optimisés avec suivi du temps
    class TimeCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.times = []
            self.epoch_times = []
            self.training_start_time = None

        def on_train_begin(self, logs=None):
            self.training_start_time = time.time()
            logging.info("Début de l'entraînement")

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            self.times.append(time.time() - self.training_start_time)
            
            # Estimation du temps restant
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = EPOCHS - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            
            logging.info(f"Epoch {epoch + 1}/{EPOCHS} - Temps: {epoch_time:.2f}s")
            logging.info(f"Accuracy: {logs['accuracy']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")
            logging.info(f"Temps moyen par epoch: {avg_epoch_time:.2f}s")
            logging.info(f"Temps estimé restant: {estimated_remaining_time/60:.2f} minutes")

    # Callbacks optimisés
    time_callback = TimeCallback()
    callbacks = [
        time_callback,
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.000001,
            verbose=1
        )
    ]

    # Nettoyage de la mémoire avant l'entraînement
    tf.keras.backend.clear_session()

    # Entraînement du modèle
    logging.info("Début de l'entraînement...")
    start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        max_queue_size=10
    )

    total_time = time.time() - start_time
    logging.info(f"Entraînement terminé en {total_time/60:.2f} minutes")
    logging.info(f"Temps moyen par epoch: {np.mean(time_callback.epoch_times):.2f}s")

    # Nettoyage de la mémoire GPU
    tf.keras.backend.clear_session()

    # Évaluation du modèle
    print("\nÉvaluation du modèle sur l'ensemble de validation:")
    results = model.evaluate(validation_generator)
    print(f"Loss: {results[0]}")
    print(f"Accuracy: {results[1]}")
    print(f"Precision: {results[2]}")
    print(f"Recall: {results[3]}")

    # Prédictions pour la matrice de confusion
    y_pred = model.predict(validation_generator)
    y_pred = (y_pred > 0.5).astype(int)
    y_true = validation_generator.classes

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatrice de confusion:")
    print(cm)

    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred))

    # Visualisation de l'historique d'entraînement
    def plot_training_history(history):
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Learning Rate
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('assets/training_history_improved.png')
        plt.close()

    plot_training_history(history)

if __name__ == '__main__':
    main() 