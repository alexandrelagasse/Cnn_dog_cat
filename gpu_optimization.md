# Optimisation de la RTX 4060 pour l'Entraînement de Modèles

## 1. Configuration TensorFlow pour la RTX 4060

Pour maximiser les performances de votre RTX 4060, il faut d'abord configurer TensorFlow correctement :

```python
import tensorflow as tf

# Vérifier que TensorFlow utilise bien le GPU
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# Configuration de la mémoire GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Permettre la croissance dynamique de la mémoire GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Limiter la mémoire GPU à 90% pour éviter les problèmes
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*9)]
        )
    except RuntimeError as e:
        print(e)
```

## 2. Optimisations du Code

### Augmentation du Batch Size
- La RTX 4060 a 8GB de VRAM, ce qui permet d'utiliser des batch sizes plus grands
- Recommandé : batch_size = 128 ou 256

### Utilisation de Mixed Precision
```python
# Activer le mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## 3. Paramètres Recommandés pour la RTX 4060

### Architecture du Modèle
- Augmenter le nombre de filtres dans les couches de convolution
- Utiliser des couches plus profondes
- Batch size : 128-256
- Image size : 256x256 ou 320x320

### Hyperparamètres d'Optimisation
```python
# Learning rate plus élevé grâce au batch size plus grand
initial_learning_rate = 0.001

# Optimiseur avec momentum
optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_learning_rate,
    beta_1=0.9,
    beta_2=0.999
)
```

## 4. Monitoring des Performances GPU

```python
# Ajouter ces callbacks pour monitorer l'utilisation GPU
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir='logs/gpu',
        profile_batch='500,520'  # Profiler les batches 500 à 520
    ),
    tf.keras.callbacks.ProgbarLogger(count_mode='steps')
]
```

## 5. Conseils Supplémentaires

1. **Gestion de la Mémoire**
   - Nettoyer régulièrement la mémoire GPU
   - Utiliser `tf.keras.backend.clear_session()` entre les entraînements

2. **Optimisation du Système**
   - Mettre à jour les drivers NVIDIA
   - Désactiver les programmes en arrière-plan
   - Utiliser le mode "Performance" dans les paramètres Windows

3. **Cooling**
   - Assurer une bonne ventilation du PC
   - Surveiller les températures du GPU (idéalement < 80°C)

## 6. Exemple de Code Optimisé

```python
# Configuration GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Mixed Precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Paramètres optimisés
BATCH_SIZE = 128
IMG_SIZE = (256, 256)

# Data augmentation avec batch size plus grand
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Optimiseur avec paramètres ajustés
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

## 7. Vérification des Performances

Pour vérifier que tout fonctionne correctement :

```python
# Vérifier l'utilisation GPU
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# Vérifier la mémoire utilisée
print("Mémoire GPU utilisée:", tf.config.experimental.get_memory_info('GPU:0'))
```

Ces optimisations devraient permettre d'utiliser votre RTX 4060 à son plein potentiel pour l'entraînement de modèles de deep learning. 