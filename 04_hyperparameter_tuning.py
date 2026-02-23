print("="*50)
print("PHASE 7: HYPERPARAMETER TUNING")
print("="*50)

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

print("Random Forest: Baseline vs Tuned Comparison")
print("=" * 60)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
train_df = pd.read_csv("train_processed.csv")
feature_cols = joblib.load("feature_columns.pkl")

X = train_df[feature_cols]
y = train_df["RUL"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
#  BASELINE MODEL
# --------------------------------------------------
print("\nTraining Baseline Random Forest...")

baseline_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

baseline_rf.fit(X_train, y_train)
baseline_pred = baseline_rf.predict(X_val)

baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))

print(f"Baseline RMSE: {baseline_rmse:.2f} cycles")

# --------------------------------------------------
#  TUNED MODEL
# --------------------------------------------------
print("\nStarting Hyperparameter Tuning...")

param_grid = {
    "n_estimators": randint(300, 800),
    "max_depth": randint(8, 25),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=50,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
tuned_pred = best_rf.predict(X_val)

tuned_rmse = np.sqrt(mean_squared_error(y_val, tuned_pred))

print(f"Tuned RMSE: {tuned_rmse:.2f} cycles")

# --------------------------------------------------
# Improvement Calculation
# --------------------------------------------------
improvement = ((baseline_rmse - tuned_rmse) / baseline_rmse) * 100

print("\nRandom Forest:")
print(f"  - Original: {baseline_rmse:.2f} cycles")
print(f"  - Tuned:    {tuned_rmse:.2f} cycles")
print(f"  - Improvement: {improvement:.1f}%")

# Save best model
joblib.dump(best_rf, "random_forest_tuned.pkl")
print("\nModel saved as: random_forest_tuned.pkl")

# MODEL 2: XGBoost Hyperparameter Tuning
print("\n" + "="*50)
print("TUNING XGBOOST")
print("="*50)

print("\n Original XGBoost Performance:")
original_xgb = pd.read_csv(r'C:\Users\Saksh\Downloads\model_comparison.csv')
original_xgb_rmse = original_xgb[original_xgb['Model'] == 'XGBoost']['Val RMSE'].values[0]
print(f"   RMSE: {original_xgb_rmse:.2f} cycles")

# Define parameter space
xgb_param_dist = {
    'n_estimators': randint(50, 100),
    'max_depth': randint(5, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5)
}

print("\n Parameter search space:")
for param, dist in xgb_param_dist.items():
    print(f"   {param}: {dist}")

# Initialize model
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Randomized search
print("\n Starting randomized search (20 iterations)...")
print("This will take 5-10 minutes...\n")

xgb_random = RandomizedSearchCV(
    xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=20,
    scoring=rmse_score,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X, y)

# Best parameters
print("\n Best XGBoost parameters found:")
for param, value in xgb_random.best_params_.items():
    print(f"   {param}: {value}")

tuned_xgb_rmse = -xgb_random.best_score_
improvement_xgb = ((original_xgb_rmse - tuned_xgb_rmse) / original_xgb_rmse) * 100

print(f"\n XGBoost Results:")
print(f"   Original RMSE: {original_xgb_rmse:.2f} cycles")
print(f"   Tuned RMSE: {tuned_xgb_rmse:.2f} cycles")
print(f"   Improvement: {improvement_xgb:.1f}%")

# Save tuned model
joblib.dump(xgb_random.best_estimator_, r'C:\Users\Saksh\Downloads\xgboost_tuned.pkl')
print(r"\n Saved: C:\Users\Saksh\Downloads\xgboost_tuned.pkl'")

# MODEL 3: LightGBM Hyperparameter Tuning
print("\n" + "="*50)
print("TUNING LIGHTGBM")
print("="*50)

print("\n Original LightGBM Performance:")
original_lgb_rmse = original_xgb[original_xgb['Model'] == 'LightGBM']['Val RMSE'].values[0]
print(f"   RMSE: {original_lgb_rmse:.2f} cycles")

# Define parameter space
lgb_param_dist = {
    'n_estimators': randint(50, 100),
    'max_depth': randint(5, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_samples': randint(10, 50),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

print("\n Parameter search space:")
for param, dist in lgb_param_dist.items():
    print(f"   {param}: {dist}")

# Initialize model
lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

# Randomized search
print("\n Starting randomized search (20 iterations)...")
print("This will take 5-10 minutes...\n")

lgb_random = RandomizedSearchCV(
    lgb_model,
    param_distributions=lgb_param_dist,
    n_iter=20,
    scoring=rmse_score,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

lgb_random.fit(X, y)

# Best parameters
print("\n Best LightGBM parameters found:")
for param, value in lgb_random.best_params_.items():
    print(f"   {param}: {value}")

tuned_lgb_rmse = -lgb_random.best_score_
improvement_lgb = ((original_lgb_rmse - tuned_lgb_rmse) / original_lgb_rmse) * 100

print(f"\n LightGBM Results:")
print(f"   Original RMSE: {original_lgb_rmse:.2f} cycles")
print(f"   Tuned RMSE: {tuned_lgb_rmse:.2f} cycles")
print(f"   Improvement: {improvement_lgb:.1f}%")

# Save tuned model
joblib.dump(lgb_random.best_estimator_, r'C:\Users\Saksh\Downloads\lightgbm_tuned.pkl')
print(r"\n Saved: 'C:\Users\Saksh\Downloads\lightgbm_tuned.pkl'")

# STEP 3: Save Tuning Results
print("\n" + "="*50)
print("SAVING TUNING RESULTS")
print("="*50)

tuning_results = pd.DataFrame({
    'Model': ['XGBoost (Original)', 'XGBoost (Tuned)', 'LightGBM (Original)', 'LightGBM (Tuned)'],
    'RMSE': [original_xgb_rmse, tuned_xgb_rmse, original_lgb_rmse, tuned_lgb_rmse],
    'Improvement': [0, improvement_xgb, 0, improvement_lgb]
})

print("\n" + tuning_results.to_string(index=False))

tuning_results.to_csv(r'C:\Users\Saksh\Downloads\hyperparameter_tuning_results.csv', index=False)
print(r"\n Saved: 'C:\Users\Saksh\Downloads\hyperparameter_tuning_results.csv'")

# Save best parameters
best_params = {
    'XGBoost': xgb_random.best_params_,
    'LightGBM': lgb_random.best_params_
}

import json
with open(r'C:\Users\Saksh\Downloads\best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=4, default=str)
print(r" Saved: C:\Users\Saksh\Downloads\best_hyperparameters.json'")

# SUMMARY
print("\n" + "="*50)
print("PHASE 7 COMPLETE! ")
print("="*50)
print(f"\n Hyperparameter Tuning Summary:")

print("\nRandom Forest:")
print(f"  - Original: {baseline_rmse:.2f} cycles")
print(f"  - Tuned:    {tuned_rmse:.2f} cycles")
print(f"  - Improvement: {improvement:.1f}%")
print(f"\nXGBoost:")
print(f"  - Original: {original_xgb_rmse:.2f} cycles")
print(f"  - Tuned: {tuned_xgb_rmse:.2f} cycles")
print(f"  - Improvement: {improvement_xgb:.1f}%")
print(f"\nLightGBM:")
print(f"  - Original: {original_lgb_rmse:.2f} cycles")
print(f"  - Tuned: {tuned_lgb_rmse:.2f} cycles")
print(f"  - Improvement: {improvement_lgb:.1f}%")
print(f"\nFiles created:")
print(f"1. xgboost_tuned.pkl")
print(f"2. lightgbm_tuned.pkl")
print(f"3. hyperparameter_tuning_results.csv")
print(f"4. best_hyperparameters.json")
print(f"\n Next: Run Phase 8 (Final Comprehensive Comparison)")
print("="*50)

