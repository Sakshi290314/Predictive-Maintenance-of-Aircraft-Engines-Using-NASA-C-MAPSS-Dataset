import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("Phase 2: Data Preprocessing")
print("-"* 50)
# Step 1: Load raw data
print("\n Step 1: Loading training data...")
columns = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
columns += [f'sensor_{i}' for i in range(1, 22)]

train_df = pd.read_csv(r'C:\Users\Saksh\Downloads\CMAPSSData\train_FD001.txt', sep='\s+', header=None, 
                        names=columns, index_col=False)
train_df = train_df.dropna(axis=1)

print(f"Data loaded successfully")
print(f"Shape: {train_df.shape}")
print(f"Engines: {train_df['unit_id'].nunique()}")


# Step 2: Calculate Remaining Useful Life (RUL)
print("\n Step 2: RUL calculating (Target variable..)")
print("RUL = max_cycle - current_cycle for each engine")

# Find the maximum operating cycle (failure point) for each engine
max_cycle= train_df.groupby('unit_id')['time_cycles'].max().reset_index()
max_cycle.columns= ['unit_id', 'max_cycle']

# Merge max cycle information with each measurement
train_df= train_df.merge(max_cycle, on='unit_id', how='left')

# Calculate RUL: how many cycles remain until failure
train_df['RUL']= train_df['max_cycle'] - train_df['time_cycles']

print(f"RUL calculated successfully")
print(f"RUL range: {train_df['RUL'].min()} to {train_df['RUL'].max()} cycles")
print("\nExample - Engine 1 (first 10 measurements):")
print(train_df[train_df['unit_id']==1][['unit_id', 'time_cycles', 'max_cycle', 'RUL']].head(10))

# Step 3: Remove low-variance sensors
print("\nStep 3: Identifying and removing low-variance sensors...")
print("Sensors that don't change provide no predictive value")

sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
sensor_variance = train_df[sensor_cols].var()

print("\nSensor variances (sorted):")
for sensor, var in sensor_variance.sort_values().items():
    print(f"  {sensor}: {var:.6f}")

# Remove sensors with very low variance (threshold < 0.001)
threshold = 0.001
useless_sensors = sensor_variance[sensor_variance < threshold].index.tolist()

print(f"\nRemoving {len(useless_sensors)} low-variance sensors: {useless_sensors}")
train_df = train_df.drop(columns=useless_sensors)

# Update the list of useful sensors
sensor_cols = [col for col in sensor_cols if col not in useless_sensors]
print(f"Keeping {len(sensor_cols)} useful sensors for modeling")

# Step 4: Analyze sensor correlations with RUL
print("\nStep 4: Analyzing sensor correlations with RUL...")
print("Higher correlation = better predictor of remaining useful life")

correlations = train_df[sensor_cols + ['RUL']].corr()['RUL'].drop('RUL').sort_values()

print("\nSensor correlations with RUL:")
for sensor, corr in correlations.items():
    print(f"  {sensor}: {corr:.4f}")

# Create correlation visualization
plt.figure(figsize=(10, 6))
correlations.plot(kind='barh', color='steelblue')
plt.title('Sensor Correlation with RUL', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig(r'C:\Users\Saksh\Downloads\sensor_correlations.png', dpi=300)
print(r"\nCorrelation chart saved to 'C:\Users\Saksh\Downloads\sensor_correlations.png'")
plt.close()


# Step 5: Normalize features
print("\nStep 5: Normalizing features to [0, 1] range...")
print("Normalization required for neural networks and improves ML performance")

feature_cols = ['setting_1', 'setting_2', 'setting_3'] + sensor_cols

print(f"Normalizing {len(feature_cols)} features (3 settings + {len(sensor_cols)} sensors)")
# Create and fit MinMaxScaler
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

print("Normalization complete")
print("\nFeature ranges after normalization:")
print(train_df[feature_cols].describe().loc[['min', 'max']])

# Step 6: Save processed data and preprocessing objects
print("\nStep 6: Saving processed data and artifacts...")

# Save processed dataframe
train_df.to_csv('train_processed.csv', index=False)
print("Saved: 'train_processed.csv'")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("Saved: 'scaler.pkl'")

# Save feature column names
joblib.dump(feature_cols, 'feature_columns.pkl')
print("Saved: 'feature_columns.pkl'")

# Summary
print("\n" + "=" * 50)
print("Phase 2 Complete!")
print("=" * 50)

print(f"\nDataset Summary:")
print(f"  Total samples: {len(train_df):,}")
print(f"  Unique engines: {train_df['unit_id'].nunique()}")
print(f"  Total features: {len(feature_cols)}")
print(f"  Sensors removed: {len(useless_sensors)}")
print(f"  Sensors kept: {len(sensor_cols)}")

print(f"\nFiles created in current project folder:")
print(f"  1. train_processed.csv")
print(f"  2. scaler.pkl")
print(f"  3. feature_columns.pkl")
print(f"  4. sensor_correlations.png")

print(f"\nNext step: Build and train machine learning models 🚀")
print("=" * 50)
