# Brain Tumor MRI Classification

This project implements a Convolutional Neural Network (CNN) for classifying brain tumor MRI images into four categories. The model is built using TensorFlow and Keras, and trained on the Brain Tumor MRI Dataset from Kaggle.

## Dataset

The dataset used in this project is the Brain Tumor MRI Dataset from Kaggle. It contains MRI scans categorized into four classes:

- Glioma
- Meningioma 
- No tumor
- Pituitary tumor

The dataset is organized into Training and Testing directories, each containing subdirectories for the four classes.

## Project Structure

project/
├── scans/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── logs/
├── cnn_classififer.py
└── README.md

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- Matplotlib

You can install the required packages using:

```bash
pip install tensorflow scikit-learn matplotlib tensorboard
```

## Usage

1. Download the dataset from Kaggle and place it in the scans/ directory with the structure shown above.

2. Run the Python script:

```bash
python brain_tumor_classification.py
```

3. The script will:
   - Load and preprocess the training and testing images
   - Build the CNN model
   - Train the model for 10 epochs
   - Display training and validation accuracy plots
   - Log training metrics to TensorBoard

4. View TensorBoard logs:
```bash
tensorboard --logdir logs/fit
```
## Results

The model training progress is visualized through:
- Accuracy and loss plots during training
- TensorBoard integration for detailed metrics tracking
- Validation accuracy measurement after each epoch

## Future Improvements

- Data augmentation to improve generalization
- Transfer learning with pre-trained models
- Hyperparameter tuning
- More sophisticated evaluation metrics
- Class imbalance handling
