# Traffic Sign Classification CNN

This project implements a Convolutional Neural Network (CNN) to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is designed to recognize 43 traffic sign classes with high accuracy, leveraging data preprocessing, augmentation, and advanced training techniques.

## Features

- **Data Conversion**: Converts raw .ppm images into pickle files for efficient loading.
- **Data Preprocessing**: Grayscale conversion, histogram equalization, and normalization.
- **Data Augmentation**: Flipping and balancing techniques to address class imbalances.
- **CNN Architecture**: Customizable with 3 convolutional layers, max pooling, dropout, and fully connected layers.
- **Training Enhancements**: Early stopping, L2 regularization, and optional learning rate decay.
- **Prediction**: Classifies new images with top-5 prediction visualization.
- **Visualization Tools**: Dataset exploration, class distribution, preprocessing effects, augmentation samples, and t-SNE plots.

## Project Structure

```
├── convert_data.py       # Converts raw GTSRB .ppm images to pickle files
├── config.json           # Configuration file for settings
├── data_loader.py        # Loads pickled data and traffic sign names
├── data_visualizer.py    # Visualizes dataset samples and distributions
├── data_processor.py     # Preprocesses and augments data
├── main.py               # Main script to run training or prediction
├── model_builder.py      # Defines the CNN architecture
├── predictor.py          # Predicts traffic sign classes for custom images
├── requirements.txt      # Lists Python dependencies
├── trainer.py            # Manages model training with callbacks
├── data/                 # Dataset directory
│   ├── signnames.csv     # Traffic sign class labels
│   ├── test.p            # Test dataset
│   ├── train.p           # Training dataset
│   ├── train_balanced.p  # Balanced training dataset
│   ├── train_extended.p  # Extended training dataset
│   ├── custom/           # Custom images for prediction
│   ├── test/             # Raw test images
│   └── train/            # Raw training images (subfolders by class ID)
├── models/               # Saved model weights and training logs
└── utils/                # Utility functions
    └── AugmentedSignsBatchIterator.py  # Data augmentation utility
```

## Installation

1. **Clone the Repository**:
```bash
git clone <repository-url>
cd TrafficSignClassificationCNN
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```
Ensure you have TensorFlow 2.x installed. For macOS with Apple Silicon, tensorflow-macos and tensorflow-metal are recommended for GPU support.

3. **Prepare the Dataset**:
   - Download the GTSRB dataset from Kaggle.
   - Place training images in data/train/ (subfolders named by class ID, e.g., 00000).
   - Place test images in data/test/Images/.
   - Ensure signnames.csv is in data/.

4. **Convert Data (optional)**:
```bash
python convert_data.py
```
This generates data/train.p and data/test.p from raw images.

## Usage

### Training the Model

Train the model with default settings:

```bash
python main.py --mode train
```
- Loads train.p and test.p.
- Preprocesses and balances the dataset (saved as train_balanced.p).
- Trains the CNN and saves weights in models/.

Use `--rebuild-datasets` to regenerate balanced datasets:
```bash
python main.py --mode train --rebuild-datasets
```

### Predicting on New Images

Classify custom images:

```bash
python main.py --mode predict
```
- Place .png images in data/custom/.
- Outputs top-5 predictions with visualizations.

### Visualization Tools

Run data_visualizer.py to explore the dataset:

```bash
python data_visualizer.py
```

Options:
- **Random Samples**: View random images from train.p.
- **Class Distribution**: Plot sample counts per class.
- **Preprocessing Comparison**: Compare original vs. preprocessed images.
- **Augmented Samples**: Show original vs. augmented images.
- **t-SNE Visualization**: Plot a 2D t-SNE projection of features.

Follow the prompts to select an option and specify parameters.

## Model Architecture

The CNN consists of:

- **Input**: 32x32x1 grayscale images.
- **Conv1**: 5x5 kernel, 32 filters, ReLU, max pooling (2x2), 10% dropout.
- **Conv2**: 5x5 kernel, 64 filters, ReLU, max pooling (2x2), 20% dropout.
- **Conv3**: 5x5 kernel, 128 filters, ReLU, max pooling (2x2), 30% dropout.
- **FC1**: 1024 units, ReLU, 50% dropout.
- **Output**: 43 units, softmax.

## Training Parameters

- **Learning Rate**: 0.001 (training), 0.0001 (prediction).
- **L2 Regularization**: Enabled (λ=0.0001).
- **Early Stopping**: Enabled (patience=100).
- **Batch Size**: 128.
- **Max Epochs**: 200.

## Data Preparation

- **Conversion**: convert_data.py resizes .ppm images to 32x32 and saves them as RGB pickle files.
- **Preprocessing**: Converts to grayscale, applies histogram equalization, and normalizes to [0,1].
- **Augmentation**:
  - **Flipping**: Extends dataset using horizontal/vertical flips for specific classes.
  - **Balancing**: Duplicates images with augmentation to balance classes (up to 2x original size).

## Requirements

See requirements.txt:
- numpy
- matplotlib
- tensorflow-macos (or tensorflow for non-Mac)
- tensorflow-metal (for Apple Silicon GPUs)
- pandas
- scikit-learn
- scikit-image
- seaborn

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [German Traffic Sign Recognition Benchmark (GTSRB) dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- TensorFlow, scikit-learn, and scikit-image libraries.
- Data augmentation techniques inspired by various research papers in computer vision:
  - [Navoshta](https://navoshta.com/traffic-signs-classification/)
  - [Victor Zhou](https://victorzhou.com/blog/intro-to-cnns-part-1/)