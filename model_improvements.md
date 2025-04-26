# CNN Dog/Cat Model Improvements

## Improved Model Architecture

### Model Structure
- 5 convolutional blocks with BatchNormalization and Dropout
- Dense layers with L2 regularization
- Use of MaxPooling2D for dimensionality reduction
- ReLU activation function for hidden layers
- Sigmoid activation function for the output layer

### Layer Configuration
1. **Block 1**:
   - Conv2D(128, (3,3))
   - BatchNormalization
   - MaxPooling2D
   - Dropout(0.25)

2. **Block 2**:
   - Conv2D(256, (3,3))
   - BatchNormalization
   - MaxPooling2D
   - Dropout(0.25)

3. **Block 3**:
   - Conv2D(512, (3,3))
   - BatchNormalization
   - MaxPooling2D
   - Dropout(0.25)

4. **Block 4**:
   - Conv2D(512, (3,3))
   - BatchNormalization
   - MaxPooling2D
   - Dropout(0.25)

5. **Block 5**:
   - Conv2D(512, (3,3))
   - BatchNormalization
   - MaxPooling2D
   - Dropout(0.25)

6. **Dense Layers**:
   - Dense(2048) + BatchNormalization + Dropout(0.5)
   - Dense(1024) + BatchNormalization + Dropout(0.5)
   - Dense(1, activation='sigmoid')

## Optimizations

### Data Augmentation
- Rotation: ±40 degrees
- Horizontal/vertical shift: 20%
- Zoom: ±20%
- Horizontal/vertical flip
- Brightness adjustment: ±20%
- Channel shift: ±20

### GPU Configuration
- Allocated GPU memory: 8GB
- Batch size: 24
- Mixed precision: mixed_float16

### Callbacks
- EarlyStopping (patience=15)
- ModelCheckpoint (save best model)
- ReduceLROnPlateau (reduce learning rate)
- TimeCallback (track training time)

### Training Parameters
- Image size: 224x224
- Initial learning rate: 0.001
- Number of epochs: 50
- Optimizer: Adam
- Loss: binary_crossentropy

## Results

### Final Metrics
- Accuracy: 89.97%
- Precision: 90.41%
- Recall: 89.43%
- Loss: 0.3704

### Confusion Matrix
```
[[915  96]
 [107 905]]
```

### Training Time
- Average time per epoch: ~124.57s
- Estimated total time: ~101.73 minutes

## Possible Future Improvements
1. Increase dataset size
2. Test other architectures (ResNet, EfficientNet)
3. Hyperparameter optimization
4. Use transfer learning
5. Implement ensembling

# CNN Dog/Cat Model Test Results

## Test on 26/04/2025

### Tested Images:

1. **c.jpeg**
   - Prediction: Cat
   - Confidence: 99.96%
   - Result: ✅ Correct
   - [Link to image](test_images/c.jpeg)

2. **OIP.jpeg**
   - Prediction: Dog
   - Confidence: 95.51%
   - Result: ✅ Correct
   - [Link to image](test_images/OIP.jpeg)

3. **R.jpeg**
   - Prediction: Cat
   - Confidence: 99.92%
   - Result: ✅ Correct
   - [Link to image](test_images/R.jpeg)

### Results Analysis

- **Success rate**: 100% (3/3 images correctly classified)
- **Average confidence**: 98.46%
- **Strengths**:
  - Very high confidence in predictions (>95% for all images)
  - Perfect classification on the test set
  - Particularly strong performance on cats with confidence >99%

### Model Parameters
- Input image size: 224x224 pixels
- Normalization: Division by 255
- Supported image formats: .jpg, .jpeg, .png

### Next Possible Improvements
1. Test with a larger set of images
2. Add more difficult cases (blurry, poorly framed images, etc.)
3. Test with images containing both dogs and cats
4. Add performance metrics (inference time, memory usage) 