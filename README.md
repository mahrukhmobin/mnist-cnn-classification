# MNIST Handwritten Digit Classification using CNN

A deep learning project that builds and trains a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset, benchmarked against traditional ML models.

---

## Project Overview

This project covers the complete deep learning workflow — from data preprocessing and visualization to model training, evaluation, and comparison with baseline models.

**Dataset:** MNIST — 60,000 training images, 10,000 test images (28x28 grayscale digits, 0-9)

---

## Key Highlights

- Visualized sample digits to understand the dataset distribution
- Preprocessed images with normalization and reshaping for CNN input
- Applied data augmentation (rotation, shift, zoom) to improve generalization
- Trained CNN with Early Stopping and Model Checkpoint callbacks
- Monitored accuracy and loss across training and validation sets
- Benchmarked CNN against Logistic Regression, Decision Tree, and Random Forest
- Evaluated final model using confusion matrix and classification report

---

## CNN Architecture

```
Input (28x28x1)
   ↓
Conv2D (32 filters, 3x3, ReLU)
   ↓
MaxPooling2D (2x2)
   ↓
Conv2D (64 filters, 3x3, ReLU)
   ↓
MaxPooling2D (2x2)
   ↓
Flatten
   ↓
Dense (128, ReLU)
   ↓
Dropout (0.5)
   ↓
Dense (10, Softmax)
```

---

## Results

| Model | Validation Accuracy |
|-------|-------------------|
| Logistic Regression | ~92% |
| Decision Tree | ~87% |
| Random Forest | ~97% |
| CNN (tuned) | ~99.4% |

CNN outperformed all baseline models with a test accuracy of **99.4%**.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | CNN model building and training |
| Scikit-learn | Baseline ML models and evaluation |
| Matplotlib & Seaborn | Visualizations and confusion matrix |
| NumPy | Data manipulation |

---

## Project Structure

```
mnist-cnn-classification/
│
├── DL and CV.py            # Main script
├── best_mnist_model.h5     # Saved best CNN model
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/mahrukhmobin/mnist-cnn-classification.git

# 2. Install dependencies
pip install tensorflow scikit-learn matplotlib seaborn numpy

# 3. Run the script
python "DL and CV.py"
```

---

## Learnings

- Designing and training CNN architectures with Keras
- Using data augmentation to reduce overfitting
- Implementing Early Stopping and Model Checkpoint for efficient training
- Comparing deep learning vs traditional ML models on image data
- Interpreting confusion matrices and classification reports

---

*Built by [Mahrukh Mobin](https://github.com/mahrukhmobin) — Computer Engineering Student @ UET Lahore*
