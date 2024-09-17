# Currency-Fraud-Detection-Model

## Project Overview

This project aims to classify banknotes as genuine or forged using various machine learning models. The dataset used consists of images taken from both real and forged banknote-like specimens. Key features were extracted from these images using Wavelet Transform, and several classification algorithms were applied, including Perceptron, Support Vector Machine (SVM), and Gaussian Naive Bayes (GaussianNB).

## Data

### Collection

- **Source**: Images were captured using an industrial camera designed for print inspection.
- **Image Specs**: 400x400 pixels, gray-scale with a resolution of 660 dpi.
- **Digitization Process**: Due to the object lens and distance to the inspected object, images were digitized with precise clarity for feature extraction.
  
### Feature Extraction

Wavelet Transform was applied to the images to extract key statistical features. The following attributes were used for model training:

- **Variance**: Wavelet Transformed image.
- **Skewness**: Skewness of the Wavelet Transformed image.
- **Kurtosis**: Kurtosis of the Wavelet Transformed image.
- **Entropy**: Entropy of the image.

These features capture the visual characteristics of the banknotes, allowing the models to distinguish between genuine and counterfeit notes.

This data was provided by [UCI](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

## Machine Learning 

Multiple machine learning models were used in this project, including (but not limited to) the following:

1. **Perceptron**: A single-layer neural network used for binary classification tasks.
2. **Support Vector Machine (SVM)**: A powerful classification algorithm that finds the optimal hyperplane for separating classes.
3. **Gaussian Naive Bayes (GaussianNB)**: A probabilistic classifier based on applying Bayes’ theorem with the assumption of normal distribution.


## Workflow

1. **Data Loading**:
   - The dataset is loaded from the `banknotes.csv` file.
   - Each row in the file is read and converted into a dictionary with the following structure:
     - **`evidence`**: A list of float values representing the features (variance, skewness, kurtosis, entropy).
     - **`label`**: A string indicating whether the note is "Authentic" or "Counterfeit".

2. **Data Preprocessing**:
   - The dataset is separated into features (`evidence`) and labels (`labels`).
   - The data is then split into training and testing groups using `train_test_split`, with 40% of the data reserved for testing.

3. **Model Training**:
   - The chosen model is trained on the training set (`X_training`, `y_training`).

4. **Model Prediction**:
   - The trained model is used to make predictions on the testing set (`X_testing`).

5. **Model Evaluation**:
   - The performance of the model is assessed using metrics such as accuracy, precision, recall, and F1-score.

6. **Results Presentation**:
   - The results of the model’s performance are printed and reviewed to determine effectiveness and identify potential areas for improvement.

## Dependencies

- **Python**: Programming language used.
- **Scikit-learn**: Library for machine learning algorithms and model evaluation:
    - `svm` for Support Vector Machine
    - `Perceptron` for Perceptron model
    - `GaussianNB` for Gaussian Naive Bayes
    - `KNeighborsClassifier` for K-Nearest Neighbors
    - `train_test_split` for splitting the data into training and testing sets
- **CSV Module**: Standard library for reading CSV files.
