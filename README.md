## 💳 Credit Card Fraud Detection

This project implements a Credit Card Fraud Detection system using a **Random Forest Classifier**. It preprocesses the dataset, trains the model, and evaluates its performance using key metrics like accuracy, confusion matrix, and classification report.

---

## 🚀 Features
- Data Preprocessing:
- Splits the dataset into training and testing sets.
- Scales features using StandardScaler for numerical stability.
 
- Random Forest Classifier:
- Trains a robust machine learning model.

- Evaluation Metrics:
- Accuracy Score
- Classification Report
- Confusion Matrix

--- 

## 🛠️ Technologies Used
- Python 🐍
- Pandas: For data manipulation.

- scikit-learn:
- Model training (Random Forest Classifier).

- Preprocessing (StandardScaler).
- Evaluation metrics (accuracy, confusion matrix, classification report).

---

## 📂 Dataset

- The dataset used in this project is a highly imbalanced credit card transaction dataset, typically containing:

- Features: Numeric values representing transaction details.

- Target Column (Class):
0: Legitimate transaction.
1: Fraudulent transaction.

- Dataset Requirements:
- Ensure the file is named creditcard.csv and located in a Dataset/ folder relative to the script.
  
--- 

## ⚙️ How It Works

- Load Dataset:
- Reads the credit card transactions dataset.
- Displays dataset structure, summary, and samples.

- Preprocessing:
- Splits data into features (X) and target (y).
- Uses train_test_split to create training and testing datasets (25% test split).
- Scales features with StandardScaler for better model performance.

- Model Training:
- Initializes a Random Forest Classifier with 100 estimators.
- Trains the model on the training set.

- Evaluation:
- Makes predictions on the test set.

- Calculates and displays:
- Accuracy Score: Overall correctness of predictions.
- Classification Report: Precision, recall, F1-score.
- Confusion Matrix: True positives, false positives, true negatives, false negatives.

---

## 📊 Results

- Accuracy Score: Measures the percentage of correctly classified transactions.
- Classification Report: Provides detailed metrics for fraud detection:
- Precision: How many predicted fraud cases are actually fraud.
- Recall: How many actual fraud cases are detected.
- F1-Score: Harmonic mean of precision and recall.
- Confusion Matrix: Helps analyze the distribution of correct and incorrect classifications.

---

## 🚀 How to Run

Clone the Repository:

git clone https://github.com/Tanish141/credit-card-fraud-detection.git
cd credit-card-fraud-detection

Install Dependencies:

pip install pandas scikit-learn

Ensure Dataset Availability:

Place the creditcard.csv file in the Dataset/ folder.

Run the Script:

python CreditCardFraudDetection.py

---

## 🏅 Key Takeaways

Learn how to preprocess highly imbalanced datasets.

Understand the implementation of a Random Forest Classifier.

Evaluate machine learning models with real-world metrics.

---

## 🤝 Contributions

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request. 🌟

---

## 📧 Contact

- For any questions or suggestions, feel free to reach out:
- Email: mrtanish14@gmail.com
- GitHub: [Your GitHub Profile](https://github.com/Tanish141)

---

## 🎉 Happy Coding!
