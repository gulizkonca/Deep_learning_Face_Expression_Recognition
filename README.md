# Deep_learning_Face_Expression_Recognition
# Face Expression Recognition using CNN + ANN

This project implements a hybrid deep learning approach to recognize human facial expressions from grayscale images. It combines a Convolutional Neural Network (CNN) for feature extraction and an Artificial Neural Network (ANN) for classification.

## Overview
- Dataset: FER2013-like structure (7 emotion classes)
- Goal: Classify images into one of the 7 basic emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Models used:
  - CNN for feature extraction from images
  - ANN for emotion classification using flattened features
- Evaluation: Accuracy, Confusion Matrix, and Classification Report

## Techniques
- TensorFlow / Keras
- ImageDataGenerator for augmentation
- CNN layers with Conv2D and MaxPooling
- ANN layers with Dense and Dropout
- Feature extraction + flattened input
- LSTM and pure ANN benchmarks included

## Results
- CNN+ANN outperformed standalone ANN and LSTM models
- Best validation accuracy: 0.3803
- Detailed performance visualized with heatmaps and plots
- Compared traditional ANN and LSTM models for grayscale face expression recognition.
CNN significantly outperformed other models with a validation accuracy over 53%, while ANN reached 38% and LSTM trailed with 23%.

## How to Run
1. Clone the repo
2. Place image dataset under `/images/train` and `/images/validation`
3. Run the notebook `Face_Expression_Recognition_Bootcamp_Final.ipynb`

## Sample Prediction
```python
img = load_img("test_image.jpg", color_mode="grayscale", target_size=(48, 48))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
pred = model_cnn.predict(img_array)
print("Predicted Emotion:", emotion_labels[np.argmax(pred)])


## Models Included
model_ann.h5

model_lstm.h5

model_cnn_ann.h5

