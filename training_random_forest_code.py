import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import joblib
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Load the Data
def load_data():
    logger.info("Loading data...")
    fundamentals = pd.read_csv(r"C:\Users\jagat\Downloads\archive\fundamentals.csv")
    prices_split_adjusted = pd.read_csv(r"C:\Users\jagat\Downloads\archive\prices-split-adjusted.csv")
    securities = pd.read_csv(r"C:\Users\jagat\Downloads\archive\securities.csv")
    
    # Display the first few rows of the loaded data for verification
    logger.info(f"Fundamentals head:\n{fundamentals.head()}")
    logger.info(f"Prices head:\n{prices_split_adjusted.head()}")
    logger.info(f"Securities head:\n{securities.head()}")
    
    logger.info("Data loaded successfully.")
    return fundamentals, prices_split_adjusted, securities

# Step 2: Preprocess Data
def preprocess_data(fundamentals, prices_split_adjusted, securities):
    logger.info("Preprocessing data...")
    scaler = StandardScaler()

    # Rename columns for consistency
    fundamentals = fundamentals.rename(columns={'Ticker Symbol': 'ticker'})
    prices_split_adjusted = prices_split_adjusted.rename(columns={'symbol': 'ticker'})

    # Ensure fundamentals have unique tickers
    fundamentals = fundamentals.drop_duplicates(subset='ticker')

    # Merge datasets
    data = pd.merge(prices_split_adjusted, fundamentals, on='ticker', how='inner')
    data = pd.merge(data, securities[['Ticker symbol', 'GICS Sector']], left_on='ticker', right_on='Ticker symbol')

    # Compute returns
    data['3M_return'] = data.groupby('ticker')['close'].pct_change(periods=63)
    data['6M_return'] = data.groupby('ticker')['close'].pct_change(periods=126)
    data['12M_return'] = data.groupby('ticker')['close'].pct_change(periods=252)

    # Drop rows with missing values
    data = data.dropna()

    # Scale selected features
    columns_to_scale = ['3M_return', '6M_return', '12M_return']
    if 'P/E' in data.columns:
        columns_to_scale.append('P/E')
    if 'ROE' in data.columns:
        columns_to_scale.append('ROE')

    scaled_features = scaler.fit_transform(data[columns_to_scale])
    scaled_column_names = [col + '_scaled' for col in columns_to_scale]
    data[scaled_column_names] = scaled_features

    logger.info(f"Data preprocessing completed. Total records: {len(data)}")
    return data

# Step 3: Label Creation
def assign_label(row):
    if row['3M_return'] > 0.2:
        return 'StrongBuy'
    elif row['3M_return'] > 0.1:
        return 'Buy'
    elif row['3M_return'] < -0.2:
        return 'StrongSell'
    elif row['3M_return'] < -0.1:
        return 'Sell'
    else:
        return 'Hold'

def label_data(data):
    logger.info("Assigning labels...")
    data['label'] = data.apply(assign_label, axis=1)
    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data['label'])
    logger.info(f"Labels assigned successfully. Classes: {label_encoder.classes_}")
    return data, label_encoder

# Step 4: Train-Test Split
def split_data(data):
    feature_columns = ['3M_return_scaled', '6M_return_scaled', '12M_return_scaled']
    if 'P/E_scaled' in data.columns:
        feature_columns.append('P/E_scaled')
    if 'ROE_scaled' in data.columns:
        feature_columns.append('ROE_scaled')

    X = data[feature_columns]
    y = data['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the splitted data to CSV files
    X_train.to_csv(r'C:\Users\jagat\Downloads\archive\X_train.csv', index=False)
    X_test.to_csv(r'C:\Users\jagat\Downloads\archive\X_test.csv', index=False)
    y_train.to_csv(r'C:\Users\jagat\Downloads\archive\y_train.csv', index=False)
    y_test.to_csv(r'C:\Users\jagat\Downloads\archive\y_test.csv', index=False)

    logger.info(f"Training data size: {len(X_train)}")
    logger.info(f"Test data size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# Step 5: Feature Selection using Recursive Feature Elimination (RFE)
def feature_selection(X, y):
    logger.info("Selecting features using Recursive Feature Elimination (RFE)...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Set n_features_to_select to 3 or fewer based on the available features
    rfe = RFE(model, n_features_to_select=min(3, X.shape[1]))  # Select top 3 or fewer features
    X_selected = rfe.fit_transform(X, y)
    
    selected_features = [col for col, support in zip(X.columns, rfe.support_) if support]
    logger.info(f"Selected Features: {selected_features}")
    
    return X_selected, selected_features

# Step 6: Handle Class Imbalance with SMOTE
def handle_class_imbalance(X_train, y_train):
    logger.info("Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: X_train shape: {X_train_smote.shape}, y_train shape: {y_train_smote.shape}")
    return X_train_smote, y_train_smote

# Step 7: Train Fusion Model with VotingClassifier
def train_fusion_model(X_train, y_train):
    logger.info("Training the fusion model...")

    # Base models with modified learning rates
    model1 = xgb.XGBClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.001,  # Reduced learning rate
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model2 = RandomForestClassifier(n_estimators=50, random_state=42)
    model3 = LogisticRegression(max_iter=500, random_state=42, C=1)  # Increased regularization (lower C)

    # Voting Classifier
    fusion_model = VotingClassifier(
        estimators=[('xgb', model1), ('rf', model2), ('lr', model3)],
        voting='soft'
    )
    fusion_model.fit(X_train, y_train)
    logger.info("Fusion model training completed.")
    return fusion_model

# Step 8: Model Evaluation
def evaluate_model(model, X_test, y_test, label_encoder):
    logger.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    logger.info(f"Classification Report:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Visualize Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(r'C:\Users\jagat\Downloads\archive\confusion_matrix.png')
    plt.close()

    return accuracy, cm

# Step 9: Save the Model
def save_model(model, destination_path):
    logger.info(f"Saving the model to {destination_path}...")
    joblib.dump(model, destination_path)
    logger.info("Model saved successfully.")

# Step 10: Plotting Graphs
def plot_graphs(data):
    logger.info("Plotting graphs...")

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['3M_return'], y=data['6M_return'], hue=data['label'])
    plt.title("3M vs 6M Returns")
    plt.savefig(r'C:\Users\jagat\Downloads\archive\scatter_plot.png')
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['label'], y=data['3M_return'])
    plt.title("Box Plot of 3M Return by Label")
    plt.savefig(r'C:\Users\jagat\Downloads\archive\box_plot.png')
    plt.close()

    # Distribution of Returns
    plt.figure(figsize=(10, 6))
    sns.histplot(data['3M_return'], kde=True)
    plt.title("Distribution of 3M Return")
    plt.savefig(r'C:\Users\jagat\Downloads\archive\distribution_3M_return.png')
    plt.close()

# Main Execution
if __name__ == "__main__":
    # Load Data
    fundamentals, prices_split_adjusted, securities = load_data()

    # Preprocess Data
    data = preprocess_data(fundamentals, prices_split_adjusted, securities)

    # Label the data
    data, label_encoder = label_data(data)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Feature selection
    X_train_selected, selected_features = feature_selection(X_train, y_train)

    # Handle class imbalance
    X_train_smote, y_train_smote = handle_class_imbalance(X_train_selected, y_train)

    # Train the fusion model
    fusion_model = train_fusion_model(X_train_smote, y_train_smote)

    # Evaluate the model
    accuracy, cm = evaluate_model(fusion_model, X_test, y_test, label_encoder)

    # Save the model
    model_save_path = r'C:\Users\jagat\Downloads\archive\fusion_model.pkl'
    save_model(fusion_model, model_save_path)

    # Plot graphs
    plot_graphs(data)

    logger.info("Process completed.")
