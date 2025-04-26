# Classification Chien/Chat avec CNN

Ce projet implémente un modèle de classification d'images pour distinguer les chiens des chats en utilisant un réseau de neurones convolutif (CNN).

## Structure du projet

```
.
├── data/               # Données d'entraînement et de test (à télécharger)
├── modele/            # Modèle entraîné
├── test_model.py      # Script pour tester le modèle
└── requirements.txt   # Dépendances Python
```

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Télécharger les données depuis Kaggle :
```bash
# Installer l'API Kaggle
pip install kaggle

# Configurer vos identifiants Kaggle
# (Placez votre fichier kaggle.json dans ~/.kaggle/)

# Télécharger le jeu de données
kaggle competitions download -c dogs-vs-cats

# Décompresser les données
unzip dogs-vs-cats.zip -d data/
```

## Utilisation

Pour tester le modèle avec une image :
```bash
python test_model.py
```

## Performance du modèle

- Accuracy : 89.97%
- Precision : 90.41%
- Recall : 89.43%

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Quiz

1. Qu'est-ce qu'un réseau neuronal convolutif (CNN) ?
    - A) Un type d'architecture de réseau neuronal utilisée pour la reconnaissance d'objets
    - B) Un type d'architecture de réseau neuronal utilisée pour le traitement du langage naturel
    - C) Un type d'architecture de réseau neuronal utilisée pour la prédiction de séries temporelles

2. Quels sont les deux types de couches principales dans un CNN ?
    - A) Couches de convolution et couches de pooling
    - B) Couches d'entrée et couches de sortie
    - C) Couches cachées et couches d'activation

3. Quel est le but des couches de convolution dans un CNN ?
    - A) Réduire la dimensionnalité des données
    - B) Extraire les caractéristiques de l'image
    - C) Classifier les images

4. Quel est le rôle des couches de pooling dans un CNN ?
    - A) Augmenter la complexité du modèle
    - B) Réduire la dimensionnalité des cartes de caractéristiques
    - C) Activer les neurones de la couche suivante

# Téléchargement du jeu de données
Pour télécharger des données depuis Kaggle, vous devez d'abord installer l'API Kaggle en utilisant pip :

```bash
pip install kaggle
```
Ensuite, vous pouvez télécharger le jeu de données des chats contre chiens :

```bash
kaggle competitions download -c dogs-vs-cats
```
# Importation des bibliothèques nécessaires
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
```

# Préparation des données

```python
Copy code
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    'data/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')
```

# Explication d'un couche de convolution

Une couche de convolution est l'une des principales composantes d'un réseau de neurones convolutif (CNN). Son travail consiste à "balayer" l'image d'entrée avec un "noyau" ou "filtre" pour extraire des caractéristiques spatiales de bas niveau, comme les bords, les textures, ou les formes.

![Convolution](assets/noyau_convolution_FR.png)

Pour imaginer comment cela fonctionne, pensez à une petite fenêtre qui glisse sur toute l'image. À chaque position, cette fenêtre (le noyau) observe la petite partie de l'image qu'elle recouvre et effectue une opération de convolution. Cette opération consiste à multiplier chaque pixel de cette partie de l'image par la valeur correspondante dans le noyau, puis à sommer toutes ces multiplications pour obtenir une seule valeur. Cette valeur unique représente une caractéristique spécifique de l'image à cette position.

# Explication d'une couche de pooling


Tout comme une couche de convolution utilise un noyau pour balayer l'image, une couche de pooling, ou de sous-échantillonnage, utilise également un "noyau" pour balayer la carte des caractéristiques produite par la couche de convolution précédente.

![Pooling](assets/pooling_kernel.png)

Cependant, l'opération effectuée par la couche de pooling est différente de celle de la convolution. Au lieu de faire une somme pondérée des valeurs de pixel, comme le fait le noyau de convolution, le noyau de pooling effectue une opération de réduction sur les valeurs qu'il recouvre.

La forme la plus courante de pooling est le max pooling, où le noyau sélectionne simplement la valeur maximale parmi toutes les valeurs qu'il recouvre.

# Construction du modèle CNN
```python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

# Entraînement du modèle

```python
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator)
```
# Évaluation du modèle

```python
score = model.evaluate(validation_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
# Exercice Bonus:

### Ajout d'images personnalisées pour l'apprentissage
Pour cet exercice, vous êtes invité à ajouter vos propres images de chats et de chiens pour l'apprentissage. Pour ce faire, vous devez les placer dans le dossier data/train respectif (c'est-à-dire data/train/dogs pour les chiens et data/train/cats pour les chats). Assurez-vous que les images sont en format JPEG et essayez d'utiliser des images de taille similaire à celles du jeu de données original.

N'oubliez pas que vous devez ajuster les chemins d'accès aux répertoires de données (train et validation) en fonction de l'endroit où vous avez stocké vos données sur votre système.


