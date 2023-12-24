# Arabic Letters Classification using EfficientNet_B7

This repository contains code for training an Arabic letters classification model using the EfficientNet_B7 architecture. The model achieves an impressive test accuracy of 97.04%. The dataset used for training is the [Final Arabic Alpha dataset](https://www.kaggle.com/competitions/arabic-letters-classification/data), and the code is implemented in PyTorch.

## Dataset

The dataset consists of Arabic letters organized into folders, with each folder representing a different class. The `MakeDataset` class is implemented to load and preprocess the dataset using PyTorch's DataLoader.

## Data Preprocessing

The images are preprocessed using the following transformations:

- Conversion to RGB format
- Grayscale conversion
- Resizing to (224, 224) pixels
- Normalization with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]

## Model Architecture

The EfficientNet_B7 model is utilized for feature extraction. The classifier's last layer is replaced with a linear layer with the correct number of output classes (65 in this case).

## Training

The model is trained using the Adam optimizer with a learning rate of 0.0001. The learning rate is adjusted using a step-wise scheduler. The training loop is designed to monitor and display training and validation losses, as well as accuracy during each epoch.

## Evaluation

The model is evaluated on a test set, and the final test accuracy is calculated. Additionally, loss and accuracy curves are plotted over the training epochs.

## Results

The trained model achieves a test accuracy of 97.04%. The loss and accuracy curves provide insights into the model's training and validation performance.

## Confusion Matrix

A confusion matrix is generated using the predictions on the test set, providing a detailed view of the model's classification performance.

## Usage

To train the model, run the provided notebook. Make sure to adjust hyperparameters and paths as needed.

## Dependencies

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- PIL

## Acknowledgments

- The EfficientNet_B7 model is from the torchvision library.


Feel free to use and modify the code for your own projects!
