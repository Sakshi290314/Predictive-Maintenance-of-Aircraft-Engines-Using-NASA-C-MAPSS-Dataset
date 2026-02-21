# Create comparison visualizations
print("\n" + "=" * 50)
print("Creating Visualizations")
print("=" * 50)

# Create figure with 3 subplots for comparing predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Random Forest
axes[0].scatter(y_val, rf_val_pred, alpha=0.5, s=10)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0].set_xlabel('True RUL (cycles)', fontsize=11)
axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[0].set_title(f'Random Forest\nRMSE: {rf_val_rmse:.2f} | R²: {rf_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# XGBoost
axes[1].scatter(y_val, xgb_val_pred, alpha=0.5, s=10, color='orange')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('True RUL (cycles)', fontsize=11)
axes[1].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[1].set_title(f'XGBoost\nRMSE: {xgb_val_rmse:.2f} | R²: {xgb_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# LightGBM
axes[2].scatter(y_val, lgb_val_pred, alpha=0.5, s=10, color='green')
axes[2].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[2].set_xlabel('True RUL (cycles)', fontsize=11)
axes[2].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[2].set_title(f'LightGBM\nRMSE: {lgb_val_rmse:.2f} | R²: {lgb_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\Saksh\Downloads\ml_predictions.png', dpi=300, bbox_inches='tight')
print(r"\nSaved: 'C:\Users\Saksh\Downloads\ml_predictions.png'")
plt.close()

# Create feature importance visualization using XGBoost
print("\nCreating feature importance chart...")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Most Important Features (XGBoost)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\Saksh\Downloads\feature_importance.png', dpi=300, bbox_inches='tight')
print(r"Saved: 'C:\Users\Saksh\Downloads\feature_importance.png'")
plt.close()

# Create model comparison table
print("\n" + "=" * 50)
print("Model Comparison Summary")
print("=" * 50)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
    'Train RMSE': [rf_train_rmse, xgb_train_rmse, lgb_train_rmse],
    'Val RMSE': [rf_val_rmse, xgb_val_rmse, lgb_val_rmse],
    'Train MAE': [rf_train_mae, xgb_train_mae, lgb_train_mae],
    'Val MAE': [rf_val_mae, xgb_val_mae, lgb_val_mae],
    'Val R²': [rf_val_r2, xgb_val_r2, lgb_val_r2]
})

print("\n" + comparison.to_string(index=False))

# Save comparison results to CSV
comparison.to_csv(r'C:\Users\Saksh\Downloads\model_comparison.csv', index=False)
print(r"\nSaved: 'C:\Users\Saksh\Downloads\model_comparison.csv'")

# Display summary
print("\n" + "=" * 50)
print("Phase 3 Complete!")
print("=" * 50)

best_model = comparison.loc[comparison['Val RMSE'].idxmin(), 'Model']
best_rmse = comparison['Val RMSE'].min()

print(f"\nBest Model: {best_model}")
print(f"Best Validation RMSE: {best_rmse:.2f} cycles")
print(f"\nFiles created:")
print(f"  1. random_forest.pkl")
print(f"  2. xgboost.pkl")
print(f"  3. lightgbm.pkl")
print(f"  4. ml_predictions.png")
print(f"  5. feature_importance.png")
print(f"  6. model_comparison.csv")
print("=" * 50)
