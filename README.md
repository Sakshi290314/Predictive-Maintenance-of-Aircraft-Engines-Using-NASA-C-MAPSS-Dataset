**Predictive Maintenance on NASA C-MAPSS Engine Dataset**

Predicting aircraft engine failures before they happen — using NASA's real-world turbofan sensor data and ensemble machine learning models (Random Forest, XGBoost, LightGBM) to estimate Remaining Useful Life (RUL) and trigger advance maintenance warnings.

**Project workflow**

<img width="655" height="504" alt="image" src="https://github.com/user-attachments/assets/e2ecaf0d-2ef8-4b60-9897-a5774e1d213a" />

**About the Dataset**

This project uses the FD001 subset — one operating condition (Sea Level) and one fault type (HPC Degradation), making it the cleanest and most focused subset for building and evaluating machine learning models.

Dataset :FD001

Engine :100 (Training) + 100 (Testing)

Condition :ONE — Sea Level

Fault :ONE — HPC (High Pressure Compressor) Degradation

Records :20,631 total sensor readings

**Understanding the Data**

Think of 100 aircraft engines, each running continuously until it breaks down. Throughout their operation, 21 sensors attached to each engine constantly measure things like temperature, pressure, fan speed, and fuel flow. 

Each row in the dataset represents one engine at one point in time:

<img width="635" height="261" alt="image" src="https://github.com/user-attachments/assets/87d08536-10bb-4b21-ac22-e45e38f51c1f" />

**What We Are Predicting**

The goal is to predict the Remaining Useful Life (RUL) — how many operational cycles an engine has left before it fails.

**Engine Life Timeline:**

<img width="498" height="253" alt="image" src="https://github.com/user-attachments/assets/1b68b671-c80a-4570-a2ce-276082f06c68" />

**Phase 1 — Data Exploration**
Before building any model, the raw sensor data was explored to understand how engine health changes over time and which sensors show meaningful degradation patterns.

**Sensor Degradation Plot**
The plot below shows sensor readings for 4 different engines over their full lifetime. Each line represents one sensor tracked from the first cycle until engine failure.

![image alt](https://github.com/Sakshi290314/Predictive-Maintenance-of-Aircraft-Engines-Using-NASA-C-MAPSS-Dataset/blob/851f2e246336fd2b25b18d396ed2f71b12e0cc81/results/engine_degradation.png)

Shows how sensor readings change as engines approach failure. Clear degradation patterns visible.

**Phase 2 — Data Preprocessing**
Step 1 — Load Raw Data

Step 2 — Calculate RUL: Created the target variable that the model will predict.

RUL = Max Cycle of Engine − Current Cycle

Engine 1 at Cycle 50 → RUL = 192 - 50 = 142 cycles left

Step 3 — Remove Useless Sensors: Sensors that never change were dropped since they add no value to the model.

Removed : sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19

Kept    : 14 sensors out of 21

Step 4 — Correlation Analysis

![image alt](https://github.com/Sakshi290314/Predictive-Maintenance-of-Aircraft-Engines-Using-NASA-C-MAPSS-Dataset/blob/a28e036be877d5aec023e2c8cc1b12396af25a8e/results/sensor_correlations.png)

Checked how strongly each remaining sensor relates to RUL. The graph above shows sensors with strong positive or negative correlation — these are the most important features for predicting engine failure.

Step 5 — Normalize Features: All 14 sensors + 3 operational settings were scaled to 0–1 range using MinMaxScaler so no feature dominates due to its measurement unit.

**Models Overview**

**Machine Learning Models**
<img width="524" height="315" alt="image" src="https://github.com/user-attachments/assets/6fe97cfb-b3ea-4df1-b781-1dfafa92d9da" />

**Feature Importance Analysis:**

<img width="1274" height="691" alt="image" src="https://github.com/user-attachments/assets/536f66c2-0d1b-4a83-a914-c5b4e110d2e1" />

Shows which sensors matter most for predictions. Temperature and pressure sensors are key.

**Machine Learning Predictions:**

<img width="1856" height="508" alt="image" src="https://github.com/user-attachments/assets/34256a3f-a608-4ff4-a3b3-7898fb84a626" />

Comparison of Random Forest, XGBoost, and LightGBM predictions vs actual RUL values.

**Hyperparameter Tuning**

Think of it like tuning a recipe:

Too much salt = bad

Too little salt = bland

Just right = perfect!

Similarly, models have "settings" we can adjust:

## ⚙️ Hyperparameter Tuning

```
XGBoost & LightGBM Model Parameters We Tuned:
-----------------------------------------------------------------
| 1. Number of Trees (n_estimators)                             |
|       Tried: 50 to 100                                        |
|       Like: How many expert opinions to combine               |
|                                                               |
| 2. Tree Depth (max_depth)                                     |
|       Tried: 5 to 15                                          |
|       Like: How many questions each tree can ask              |
|                                                               |
| 3. Learning Rate                                              |
|       Tried: 0.01 to 0.2                                      |
|       Like: How fast the model learns (slow = careful)        |
|                                                               |
| 4. Feature Sampling (colsample_bytree)                        |
|       Tried: 0.6 to 1.0                                       |
|       Like: What % of sensors to look at each time            |
-----------------------------------------------------------------

Process:
  Random Search → Try 20 different combinations
               → Pick the best performing one
               → Improves accuracy by 2-5%

Random Forest Model Parameters We Tuned
-----------------------------------------------------------------
| 1. Number of Trees (n_estimators)                             |
|       Tried: 300 to 800                                       |
|       Like: How many expert opinions to combine               |
|                                                               |
| 2. Tree Depth (max_depth)                                     |
|       Tried: 8 to 25                                          |
|       Like: How many questions each tree can ask              |
|                                                               |
| 3. Minimum Split Size (min_samples_split)                     |
|       Tried: 2 to 10                                          |
|       Like: Min data points needed to split a branch          |
|                                                               |
| 4. Minimum Leaf Size (min_samples_leaf)                       |
|       Tried: 1 to 5                                           |
|       Like: Min data points required at each leaf node        |
|                                                               |
| 5. Feature Selection (max_features)                           |
|       Tried: sqrt, log2, None (all features)                  |
|       Like: How many sensors to consider at each split        |
-----------------------------------------------------------------

Process:
  Random Search → Try 20 different combinations
               → Pick the best performing one
               → Improves accuracy by 2-5%

<img width="591" height="130" alt="image" src="https://github.com/user-attachments/assets/5f2133f4-bd20-42a2-98dd-33d88b5b1269" />]

**Use Trained Model for Predictions**

**Results Summary**

Performance Comparison**
<img width="1308" height="507" alt="image" src="https://github.com/user-attachments/assets/4a12d3c3-4418-4720-b78d-e6217df0315d" />

**Visual Performance Comparison**

**Comprehensive Model Comparison:****

<img width="1692" height="958" alt="image" src="https://github.com/user-attachments/assets/eed86c87-9ecc-4ae8-bbcb-95617cf51739" />

**Performance Improvement Chart:**

<img width="1751" height="777" alt="image" src="https://github.com/user-attachments/assets/ff14fcb9-065a-4784-95fc-ac0b61f4cb94" />

Among the machine learning models tested, XGBoost achieved the best results in terms of accuracy and error reduction.

**Project Structure**
Predictive-Maintenance-of-Aircraft-Engines-Using-NASA-C-MAPSS-Dataset/
│
├── 📂 data/
│   ├── RUL_FD001.txt                  # Ground truth RUL values
│   ├── readme.txt                     # Dataset description
│   ├── test_FD001.txt                 # Test set (raw)
│   ├── train_FD001.txt                # Training set (raw)
│   └── train_processed.csv            # Preprocessed training data
│
├── 📂 models/
│   ├── best_hyperparameters.json      # Best params from tuning
│   ├── feature_columns.pkl            # Selected feature list
│   ├── hyperparameter_tuning_results. # Full tuning logs
│   ├── lightgbm.pkl                   # LightGBM (original)
│   ├── lightgbm_tuned.pkl             # LightGBM (tuned)
│   ├── scaler.pkl                     # Feature scaler
│   ├── xgboost.pkl                    # XGBoost (original)
│   └── xgboost_tuned.pkl              # XGBoost (tuned)
│
├── 📂 results/
│   ├── FINAL_ML_COMPARISON_TABLE.csv  # Full metrics comparison
│   ├── ML_DASHBOARD.png               # 6-panel performance dashboard
│   ├── ML_IMPROVEMENT_CHART.png       # Improvement over baseline chart
│   ├── ML_RANKING_TABLE.png           # Model ranking table
│   ├── engine_degradation.png         # Engine degradation visualization
│   ├── feature_importance.png         # Feature importance plot
│   ├── ml_predictions.png             # Predicted vs actual RUL
│   ├── model_comparison.csv           # Model comparison summary
│   └── sensor_correlations.png        # Sensor correlation heatmap
│
├── 01_data_exploration.py             # EDA & sensor analysis
├── 02_data_preprocessing.py          # Feature engineering & scaling
├── 03_ml_baseline.py                 # Baseline ML model training
├── 04_hyperparameter_tuning.py       # GridSearch / tuning
├── 05_final_comparison.py            # Final evaluation & plots
│
├── LICENSE
├── README.md
└── requirements.txte

**Technical Details**

**Metrics Explained Simply**

**RMSE (Root Mean Square Error):** Average prediction error in cycles

Lower is better

Our best: 5.27 cycles (like being off by 1 day)

**MAE (Mean Absolute Error):** Average difference between prediction and reality

Lower is better

Our best: 4.09 cycles

**R² Score:** How much of the pattern does the model understand?

0 = random guessing

1 = perfect prediction

Our best: 0.9915 (99.15% accurate!)

**Real-World Impact**

**References**

**Dataset**

Saxena, A., & Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.

Link: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

**Research Papers**

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM Conference. 

**Technologies Used**

Machine Learning: Scikit-learn, XGBoost, LightGBM

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn
