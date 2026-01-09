# Machine Learning Projects

A collection of machine learning implementations exploring various algorithms and neural network architectures using Python, scikit-learn, and PyTorch.

##  Table of Contents

- [Projects Overview](#projects-overview)
- [Installation](#installation)
- [Project Details](#project-details)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Contributing](#contributing)

##  Projects Overview

This repository contains 5 Jupyter notebooks demonstrating different machine learning concepts:

1. **K-Means Clustering** - Unsupervised learning with the Iris dataset
2. **Naive Bayes Classifier** - Spam detection using text classification
3. **Simple Neural Network** - Basic MNIST digit recognition
4. **Extended Neural Network** - Enhanced MNIST classifier with training metrics
5. **Multi-Layer Perceptron (MLP)** - Deep neural network architecture

##  Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Required Libraries

```bash
pip install torch torchvision
pip install scikit-learn
pip install pandas
pip install matplotlib seaborn
pip install jupyter notebook
```

Or install all dependencies at once:

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn jupyter
```

##  Project Details

### 1. K-Means Clustering (`K_Clustering.ipynb`)

**Dataset**: Iris Dataset (150 samples, 3 species)

**Description**: Implements K-Means clustering algorithm to group iris flowers into 3 clusters based on their features (petal length, petal width, sepal length, sepal width).

**Key Features**:

- Unsupervised learning approach
- Visualization using seaborn scatter plots
- Confusion matrix comparing clusters to true labels
- Demonstrates cluster evaluation

**Usage**:

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(iris.data)
```

### 2. Naive Bayes Classifier (`Naive_Bayes.ipynb`)

**Dataset**: SMS Spam Collection

**Description**: Text classification model to detect spam messages using Multinomial Naive Bayes algorithm with bag-of-words representation.

**Key Features**:

- Text preprocessing with CountVectorizer
- Binary classification (spam/ham)
- Performance metrics with classification report
- ~99% accuracy achieved

**Usage**:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)
```

### 3. Simple Neural Network (`simpleNN.ipynb`)

**Dataset**: MNIST Handwritten Digits

**Description**: Basic 2-layer neural network built with PyTorch for digit recognition.

**Architecture**:

- Input Layer: 784 neurons (28×28 flattened images)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9)

**Usage**:

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 4. Extended Neural Network (`simpleNN_extended.ipynb`)

**Dataset**: MNIST Handwritten Digits

**Description**: Enhanced version of the simple neural network with comprehensive training loop including loss tracking and accuracy metrics.

**Improvements**:

- Multi-epoch training (10 epochs)
- Real-time loss and accuracy monitoring
- Achieved ~97% accuracy after 3 epochs

**Features**:

- Running loss calculation
- Accuracy tracking per epoch
- Progress visualization

### 5. Deep MLP Architecture (`MLP Architecture.ipynb`)

**Dataset**: MNIST Handwritten Digits

**Description**: Deeper multi-layer perceptron with 4 fully connected layers for improved feature extraction.

**Architecture**:

- Input Layer: 784 neurons
- Hidden Layer 1: 256 neurons with ReLU
- Hidden Layer 2: 128 neurons with ReLU
- Hidden Layer 3: 64 neurons with ReLU
- Output Layer: 10 neurons

**Usage**:

```python
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
```

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework for neural network implementations
- **scikit-learn**: Machine learning library for clustering and classification
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **NumPy**: Numerical computing

##  Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Maahela/Machine-Learning.git
   cd Machine-Learning
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

4. **Open any notebook** and run the cells sequentially

##  Results Summary

| Project            | Algorithm      | Dataset  | Accuracy/Performance     |
| ------------------ | -------------- | -------- | ------------------------ |
| K-Means Clustering | K-Means        | Iris     | Good cluster separation  |
| Naive Bayes        | Multinomial NB | SMS Spam | ~99% accuracy            |
| Simple NN          | 2-layer MLP    | MNIST    | Training complete        |
| Extended NN        | 2-layer MLP    | MNIST    | ~97% accuracy (3 epochs) |
| Deep MLP           | 4-layer MLP    | MNIST    | Training complete        |

##  Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.


##  Author

**Maahela**

- GitHub: [@Maahela](https://github.com/Maahela)

##  Acknowledgments

- Iris Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- MNIST Dataset: [Yann LeCun's Website](http://yann.lecun.com/exdb/mnist/)
- SMS Spam Collection: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)


