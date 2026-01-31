import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# To ensure tha we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import get_ann_model, get_regression_model

def train_pipeline():
    # 1. Load Data
    data_path = os.path.join('data', 'raw_data.csv')
    if not os.path.exists(data_path):
        print("Error: Data file not found. Please run 'python data/generate_synthetic_data.py' first.")
        return

    df = pd.read_csv(data_path)

    # 2. Select Features based on Optimal Architecture (Scenario 4 in Table 6)
    # Inputs: Q (Volume), V (Speed), PH (Heavy Vehicles), G (Gradient), D (Density), BRF (Reflection)
    feature_cols = ['Q', 'V', 'PH', 'G', 'D', 'BRF']
    target_col = 'LAeq'

    X = df[feature_cols]
    y = df[target_col]

    # 3. Preprocessing
    # Neural Networks require input scaling (0-1 range).
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Data Splitting
    # The paper uses 80% Train, 10% Validation, 10% Test.
    # Step 1: Split 80% Train, 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Step 2: Split Temp into 50% Val, 50% Test (which results in 10% total each)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Dataset Sizes - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # 5. Train ANN Model
    print("Training ANN (6-10-1 Architecture)...")
    ann = get_ann_model()
    ann.fit(X_train, y_train)

    # 6. Train Regression Model (for comparison)
    print("Training Linear Regression Model...")
    reg = get_regression_model()
    reg.fit(X_train, y_train)

    # 7. Predictions
    y_pred_ann = ann.predict(X_test)
    y_pred_reg = reg.predict(X_test)

    # 8. Evaluation
    print("\n--- RESULTS (Test Set) ---")

    mse_ann = mean_squared_error(y_test, y_pred_ann)
    r2_ann = r2_score(y_test, y_pred_ann)
    print(f"ANN Model -> MSE: {mse_ann:.4f}, R2: {r2_ann:.4f}")

    mse_reg = mean_squared_error(y_test, y_pred_reg)
    r2_reg = r2_score(y_test, y_pred_reg)
    print(f"Regression -> MSE: {mse_reg:.4f}, R2: {r2_reg:.4f}")

    # 9. Visualization (Comparison Graph)
    plt.figure(figsize=(10, 6))
    samples = range(len(y_test))
    plt.plot(samples, y_test.values, label='Measured (Target)', color='black', marker='o')
    plt.plot(samples, y_pred_ann, label='ANN Predicted', color='red', marker='x', linestyle='--')
    plt.plot(samples, y_pred_reg, label='Regression Predicted', color='blue', marker='s', linestyle=':', alpha=0.5)

    plt.title('Prediction Accuracy: ANN vs Regression vs Target')
    plt.xlabel('Sample Number')
    plt.ylabel('Equivalent Sound Level LAeq (dBA)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_pipeline()
