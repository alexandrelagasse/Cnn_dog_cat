# Dog/Cat Classification with CNN

This project implements an image classification model to distinguish between dogs and cats using a Convolutional Neural Network (CNN).

## Project Structure

```
.
├── data/               # Training and test data (to download)
├── modele/             # Trained model
├── test_model.py       # Script to test the model
└── requirements.txt    # Python dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the data from Kaggle:
```bash
# Install the Kaggle API
pip install kaggle

# Configure your Kaggle credentials
# (Place your kaggle.json file in ~/.kaggle/)

# Download the dataset
kaggle competitions download -c dogs-vs-cats

# Unzip the data
unzip dogs-vs-cats.zip -d data/
```

## Model Performance

- Accuracy: 89.97%
- Precision: 90.41%
- Recall: 89.43%

## License

This project is under the MIT license. See the LICENSE file for more details.

# Downloading the Dataset
To download data from Kaggle, you first need to install the Kaggle API using pip:

```bash
pip install kaggle
```
Then, you can download the Dogs vs Cats dataset:

```bash
kaggle competitions download -c dogs-vs-cats
```
# Importing Required Libraries
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
```

# Data Preparation

```python
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

# Explanation of a Convolutional Layer

A convolutional layer is one of the main components of a Convolutional Neural Network (CNN). Its job is to "scan" the input image with a "kernel" or "filter" to extract low-level spatial features, such as edges, textures, or shapes.

![Convolution](assets/noyau_convolution_FR.png)

To imagine how this works, think of a small window sliding over the entire image. At each position, this window (the kernel) looks at the small part of the image it covers and performs a convolution operation. This operation consists of multiplying each pixel in this part of the image by the corresponding value in the kernel, then summing all these multiplications to obtain a single value. This unique value represents a specific feature of the image at that position.

# Explanation of a Pooling Layer

Just like a convolutional layer uses a kernel to scan the image, a pooling layer (or subsampling layer) also uses a "kernel" to scan the feature map produced by the previous convolutional layer.

![Pooling](assets/pooling_kernel.png)

However, the operation performed by the pooling layer is different from that of convolution. Instead of doing a weighted sum of pixel values, as the convolution kernel does, the pooling kernel performs a reduction operation on the values it covers.

The most common form of pooling is max pooling, where the kernel simply selects the maximum value among all the values it covers.

# Building the CNN Model
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

# Training the Model

```python
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator)
```
# Model Evaluation

```python
score = model.evaluate(validation_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
# Bonus Exercise:

### Adding Custom Images for Training
For this exercise, you are invited to add your own images of cats and dogs for training. To do this, place them in the respective data/train folders (i.e., data/train/dogs for dogs and data/train/cats for cats). Make sure the images are in JPEG format and try to use images of similar size to those in the original dataset.

Remember to adjust the data directory paths (train and validation) according to where you have stored your data on your system.


