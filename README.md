# AI-Driven-Portfolio-Management-System-for-Risk-Optimization-and-Return-Maximization

## Overview
This project is a machine learning pipeline for stock market prediction. It involves data preprocessing, feature engineering, handling class imbalance, model training using an ensemble voting classifier, and evaluation. The project outputs the trained model and visualizations for insights into the data.

---

## Prerequisites

### Software Requirements:
- Python 3.7 or above
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - imbalanced-learn
  - seaborn
  - matplotlib
  - joblib

### Install Libraries:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn seaborn matplotlib joblib
```

### Files Required:
- `fundamentals.csv`
- `prices-split-adjusted.csv`
- `securities.csv`

Place these files in the directory specified in the code (e.g., `C:\Users\jagat\Downloads\archive`).

---

## Steps to Run the Code

### Step 1: Clone or Copy the Code
Ensure the script is saved as `main.py` or a similar name.

### Step 2: Execute the Code
Run the following command in the terminal:
```bash
python main.py
```

### Step 3: Outputs
- Processed datasets:
  - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- Visualizations:
  - `confusion_matrix.png`
  - `scatter_plot.png`
  - `box_plot.png`
  - `distribution_3M_return.png`
- Trained Model:
  - `fusion_model.pkl`

All outputs are saved in the `C:\Users\jagat\Downloads\archive` directory.

---

## Code Description

### Key Functions

**1. Data Loading (`load_data`):
**  
Loads and logs basic statistics of the input datasets.

2. **Data Preprocessing (`preprocess_data`)**:
   - Renames columns for consistency.
   - Merges datasets.
   - Computes new features (e.g., 3M, 6M, 12M returns).
   - Scales numerical features.

3. **Labeling (`label_data`)**:
   Assigns categorical labels (e.g., `StrongBuy`, `Buy`, `Hold`, `Sell`, `StrongSell`) based on 3-month returns.

4. **Train-Test Split (`split_data`)**:
   Splits data into training and testing sets.

5. **Feature Selection (`feature_selection`)**:
   Uses Recursive Feature Elimination (RFE) to select the top features.

6. **Class Imbalance Handling (`handle_class_imbalance`)**:
   Balances the dataset using SMOTE (Synthetic Minority Oversampling Technique).

7. **Fusion Model Training (`train_fusion_model`)**:
   Trains a soft-voting classifier using:
   - XGBoost
   - Random Forest
   - Logistic Regression

8. **Model Evaluation (`evaluate_model`)**:
   - Generates accuracy, classification report, and confusion matrix.
   - Visualizes the confusion matrix.

9. **Save Model (`save_model`)**:
   Saves the trained model as a `.pkl` file using Joblib.

10. **Graph Plotting (`plot_graphs`)**:
    - Creates scatter, box, and distribution plots to explore the data visually.

### Main Execution Workflow
1. Load the datasets.
2. Preprocess and engineer features.
3. Label data and encode labels.
4. Split data into train and test sets.
5. Perform feature selection.
6. Address class imbalance.
7. Train the ensemble model.
8. Evaluate the model and save outputs.
9. Generate visualizations.

---

## Contact
For queries or suggestions, feel free to contact the project author.

---

## License
This project is released under the MIT License.
