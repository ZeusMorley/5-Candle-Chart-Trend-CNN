# 5-Candle Chart Trend CNN

A Convolutional Neural Network (CNN) classifier for identifying market trends in candlestick charts. This project uses deep learning to classify 5-candle patterns as uptrend, downtrend, or sideways movement based solely on visual patterns.

## 📊 Project Overview

This project aims to emulate human-like pattern recognition in technical analysis by using CNNs to classify candlestick patterns. Unlike traditional charting platforms that rely on OHLC (Open, High, Low, Close) data, this approach focuses on the visual shape and size of candlesticks relative to each other.

**Author:** Zeus Morley S. Pineda  
**Institution:** Mindanao State University - Iligan Institute of Technology  
**Contact:** [zeusmorley.pineda@g.msuiit.edu.ph](mailto:zeusmorley.pineda@g.msuiit.edu.ph)

## 🎯 Objective

- **For Beginners:** A learning tool to practice recognizing candlestick trends
- **For Experienced Traders:** An aid to confirm trend classifications and improve decision-making
- **Target Accuracy:** ≥80% (aligned with real-world trading requirements)

## ✨ Features

- **Three-Class Classification:** Uptrend, Downtrend, and Sideways
- **Visual Pattern Recognition:** Uses only the shape and relative size of candles
- **High Accuracy:** Achieves 82% test accuracy with the best model
- **Lightweight Images:** 128x128 pixel grayscale images for efficient processing
- **No Data Augmentation:** Images maintain their natural vertical orientation

## 📁 Dataset

- **Total Images:** 330 (110 per class)
- **Image Size:** 128x128 pixels
- **Color Mode:** Grayscale/Black & White
- **Classes:** Up, Down, Side (uniformly distributed)
- **Source:** Screenshots from [Tradingview.com](https://www.tradingview.com/)
- **Dataset Link:** [Google Drive](https://drive.google.com/drive/folders/1iJAPnfSfsPiD6xGhPTfsca6ADNWlMlXa?usp=drive_link)

### Dataset Structure
```
dataset/
├── up/         # 110 uptrend images
├── down/       # 110 downtrend images
└── side/       # 110 sideways trend images
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or pipenv

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/5-candle-chart-trend-cnn.git
   cd 5-candle-chart-trend-cnn
   ```

2. **Install dependencies:**

   **Option A: Using pip**
   ```bash
   pip install -r requirements.txt
   ```

   **Option B: Using pipenv**
   ```bash
   pipenv install
   pipenv shell
   ```

3. **Download the dataset:**
   - Download the dataset from the [Google Drive link](https://drive.google.com/drive/folders/1iJAPnfSfsPiD6xGhPTfsca6ADNWlMlXa?usp=drive_link)
   - Extract the images into the `dataset/` folder with the structure shown above

## 💻 Usage

### Running the Best Model ⭐

**The `notebook_best.ipynb` notebook contains the optimal model with the best performance (82% test accuracy).** This is the recommended notebook to use for training and evaluation.

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

2. **Open the notebook:**
   - Navigate to and open `notebook_best.ipynb`

3. **Run the cells:**
   - Execute all cells sequentially (Cell → Run All)
   - The notebook will:
     - Load and preprocess the dataset
     - Split data into training (70%), validation (15%), and test (15%) sets
     - Build and train the CNN model
     - Evaluate performance with confusion matrix and classification report
     - Visualize misclassified examples

### Other Notebooks

- `notebook_2.ipynb` - Experimental model version 2
- `notebook_3.ipynb` - Experimental model version 3

## 🏗️ Model Architecture

The CNN model in `notebook_best.ipynb` consists of:

```
- Input Layer: 128x128x1 (grayscale images)
- Conv2D: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Conv2D: 64 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Flatten Layer
- Dense: 128 units, ReLU activation
- Dropout: 0.75 (prevent overfitting)
- Dense Output: 3 units, Softmax activation
```

**Hyperparameters:**
- Optimizer: Adam (learning rate: 0.00001)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 16
- Epochs: 20

## 📈 Performance

### Best Model Results (`notebook_best.ipynb`)

- **Test Accuracy:** 82.00%
- **Validation Accuracy:** 89.80%
- **Training Accuracy:** 93.39%

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Up    | 0.93      | 0.76   | 0.84     | 17      |
| Down  | 1.00      | 0.76   | 0.87     | 17      |
| Side  | 0.65      | 0.94   | 0.77     | 16      |

**Overall Accuracy:** 82%

### Key Insights

- The model performs best on **downtrend** patterns (100% precision)
- **Sideways** patterns have the highest recall (94%) but lower precision
- Minimal overfitting with only 3.59% accuracy gap between training and validation

## 📦 Requirements

Main dependencies:
- TensorFlow 2.18.0
- Keras 3.7.0
- NumPy 2.0.2
- Matplotlib 3.9.3
- Seaborn 0.13.2
- scikit-learn 1.5.2
- Pillow 11.0.0
- Jupyter 1.1.1

See `requirements.txt` for the complete list of dependencies.

## 🗂️ Project Structure

```
5-candle-chart-trend-cnn/
├── dataset/
│   ├── up/              # Uptrend images
│   ├── down/            # Downtrend images
│   └── side/            # Sideways trend images
├── notebook_best.ipynb  # ⭐ Best performing model
├── notebook_2.ipynb     # Experimental model v2
├── notebook_3.ipynb     # Experimental model v3
├── requirements.txt     # Python dependencies
├── Pipfile             # Pipenv configuration
├── Pipfile.lock        # Pipenv lock file
└── README.md           # This file
```

## 🎓 Academic Context

This project was developed as part of coursework at Mindanao State University - Iligan Institute of Technology. The model achieved a total score of 33/33 across evaluation criteria including data management, model training, and performance interpretation.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

**Zeus Morley S. Pineda**  
Email: [zeusmorley.pineda@g.msuiit.edu.ph](mailto:zeusmorley.pineda@g.msuiit.edu.ph)

## 🙏 Acknowledgments

- Dataset sourced from [TradingView](https://www.tradingview.com/)
- Built with TensorFlow and Keras

---

⭐ **Remember:** Use `notebook_best.ipynb` for the best results with 82% test accuracy!

