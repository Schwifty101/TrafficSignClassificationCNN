# Traffic Sign Classification CNN

A deep learning project for classifying traffic signs using Convolutional Neural Networks (CNN). This project is designed to classify German Traffic Sign Recognition Benchmark (GTSRB) dataset images with high accuracy.

## Project Overview

This project implements a CNN architecture to classify traffic signs. It includes data loading, preprocessing, model building, training, evaluation, and prediction components. The model achieves high accuracy by applying various techniques including data augmentation, early stopping, and regularization.

## Features

- Data loading from raw images or pickle files
- Advanced preprocessing pipeline including grayscale conversion and histogram equalization
- Data augmentation for balancing classes and improving generalization
- CNN architecture with configurable parameters
- Training with early stopping to prevent overfitting
- L2 regularization and dropout for model regularization
- Learning rate decay for optimization
- Email notifications for training progress and completion
- Model evaluation and prediction on new images
- Data visualization for model performance

## Project Structure

```
├── config.json           # Configuration file for email notifications
├── data_loader.py        # Utilities for loading pickle data
├── data_processor.py     # Data preprocessing and augmentation
├── evaluator.py          # Model evaluation utilities
├── main.py               # Main entry point for the application
├── model_builder.py      # CNN model architecture
├── predictor.py          # Prediction on new images
├── requirements.txt      # Project dependencies
├── trainer.py            # Model training utilities
├── data/                 # Dataset directory
│   ├── signnames.csv     # Traffic sign class labels
│   ├── test.p            # Test dataset
│   ├── train.p           # Training dataset
│   ├── train_balanced.p  # Balanced training dataset
│   ├── train_extended.p  # Extended training dataset
│   ├── test/             # Raw test images
│   └── train/            # Raw training images
├── models/               # Saved model checkpoints
└── utils/                # Utility functions
    └── AugmentedSignsBatchIterator.py  # Data augmentation utility
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TraficSignClassificationCNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure email notifications (optional):
   - Update `config.json` with your SMTP server settings.

## Usage

### Training the Model

To train the model with default parameters:

```bash
python main.py --mode train
```

The training process will:
- Load and preprocess the dataset
- Create balanced datasets for training
- Train the CNN model
- Save model checkpoints
- Generate learning curves
- Send email notifications if configured

### Evaluating the Model

To evaluate the trained model on the test dataset:

```bash
python main.py --mode evaluate
```

### Predicting on New Images

To run predictions on custom images:

```bash
# Place your images in the data/custom folder (PNG format)
python main.py --mode predict
```

## Model Architecture

The CNN architecture consists of:

- 3 convolutional layers with ReLU activation and max pooling
- Dropout after each pooling layer
- Fully connected layer with ReLU activation
- Output layer with softmax activation

### Default Parameters

```
- Convolutional Layer 1: 5x5 kernel, 32 filters, 90% dropout keep probability
- Convolutional Layer 2: 5x5 kernel, 64 filters, 80% dropout keep probability
- Convolutional Layer 3: 5x5 kernel, 128 filters, 70% dropout keep probability
- Fully Connected Layer: 1024 neurons, 50% dropout keep probability
- Learning Rate: 0.0001
- L2 Regularization: Enabled with lambda=0.0001
- Early Stopping: Enabled with patience=100
```

## Performance

The model achieves high accuracy on the GTSRB dataset. Training curves are saved in the model directory.

## Notifications

The system can send email notifications with training updates. Configure your email settings in `config.json`:

```json
{
  "logger_config": {
    "email_config": {
      "smtp_server": "smtp.example.com",
      "smtp_port": 465,
      "sender_email": "your-email@example.com",
      "receiver_email": "your-email@example.com",
      "password": "your-app-password"
    }
  }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [German Traffic Sign Recognition Benchmark (GTSRB) dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- TensorFlow and scikit-learn libraries
- Data augmentation techniques inspired by various research papers in computer vision