"""
Phase 3: Machine Learning Baseline Models

This script trains three traditional machine learning models on the preprocessed data:
1. Random Forest - Ensemble of decision trees
2. XGBoost - Gradient boosting with regularization
3. LightGBM - Fast gradient boosting variant

These models serve as baselines to compare against deep learning approaches.

Usage:
    python 03_ml_baseline.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

print("Phase 3: Machine Learning Baseline Models")
print("=" * 50)

# Load preprocessed data
print("\nLoading preprocessed data...")
train_df = pd.read_csv('train_processed.csv')
feature_cols = joblib.load('feature_columns.pkl')

print(f"Data loaded: {train_df.shape}")
print(f"Features: {len(feature_cols)}")
# Prepare features and target variable
X = train_df[feature_cols]
y = train_df['RUL']

# Split data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Training samples: {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")

# Model 1: Random Forest
print("\n" + "=" * 50)
print("Training Model 1: Random Forest")
print("=" * 50)

print("\nRandom Forest is an ensemble of decision trees that reduces overfitting...")
rf_model = RandomForestRegressor(
    n_estimators=100,        # Number of trees in the forest
    max_depth=20,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples in leaf node
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    verbose=1
)

rf_model.fit(X_train, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_val)

# Metrics
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)

rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
rf_val_mae = mean_absolute_error(y_val, rf_val_pred)
rf_val_r2 = r2_score(y_val, rf_val_pred)

print(f"\n Random Forest Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {rf_train_rmse:.2f} cycles")
print(f"  MAE:  {rf_train_mae:.2f} cycles")
print(f"  R²:   {rf_train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {rf_val_rmse:.2f} cycles")
print(f"  MAE:  {rf_val_mae:.2f} cycles")
print(f"  R²:   {rf_val_r2:.4f}")

# Save model
joblib.dump(rf_model, 'random_forest.pkl')
print("\n Model saved: 'random_forest.pkl'")

# Model 2:XGBoost
print("\n" + "=" * 50)
print("Training Model 2: XGBoost")
print("=" * 50)

print("\nXGBoost uses gradient boosting with advanced regularization techniques...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,        # Number of boosting rounds
    max_depth=10,            # Maximum tree depth
    learning_rate=0.1,       # Step size shrinkage
    subsample=0.8,           # Fraction of samples per tree
    colsample_bytree=0.8,    # Fraction of features per tree
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# Predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)

# Metrics
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
xgb_train_r2 = r2_score(y_train, xgb_train_pred)

xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
xgb_val_r2 = r2_score(y_val, xgb_val_pred)

print(f"\n XGBoost Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {xgb_train_rmse:.2f} cycles")
print(f"  MAE:  {xgb_train_mae:.2f} cycles")
print(f"  R²:   {xgb_train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {xgb_val_rmse:.2f} cycles")
print(f"  MAE:  {xgb_val_mae:.2f} cycles")
print(f"  R²:   {xgb_val_r2:.4f}")

# Save model
joblib.dump(xgb_model, 'xgboost.pkl')
print("\n Model saved: 'xgboost.pkl'")

# Model 3: LightGBM
print("\n" + "=" * 50)
print("Training Model 3: LightGBM")
print("=" * 50)

print("\nLightGBM is an efficient gradient boosting framework with fast training...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1                # Suppress training logs
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

# Predictions
lgb_train_pred = lgb_model.predict(X_train)
lgb_val_pred = lgb_model.predict(X_val)

# Metrics
lgb_train_rmse = np.sqrt(mean_squared_error(y_train, lgb_train_pred))
lgb_train_mae = mean_absolute_error(y_train, lgb_train_pred)
lgb_train_r2 = r2_score(y_train, lgb_train_pred)

lgb_val_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
lgb_val_mae = mean_absolute_error(y_val, lgb_val_pred)
lgb_val_r2 = r2_score(y_val, lgb_val_pred)

print(f"\n LightGBM Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {lgb_train_rmse:.2f} cycles")
print(f"  MAE:  {lgb_train_mae:.2f} cycles")
print(f"  R²:   {lgb_train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {lgb_val_rmse:.2f} cycles")
print(f"  MAE:  {lgb_val_mae:.2f} cycles")
print(f"  R²:   {lgb_val_r2:.4f}")

# Save model
joblib.dump(lgb_model, 'lightgbm.pkl')
print("\n Model saved: 'lightgbm.pkl'")
