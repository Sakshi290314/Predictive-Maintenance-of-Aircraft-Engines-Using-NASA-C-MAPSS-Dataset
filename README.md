**Predictive Maintenance on NASA C-MAPSS Engine Dataset**

🔧 Predicting aircraft engine failures before they happen — using NASA's real-world turbofan sensor data and ensemble machine learning models (Random Forest, XGBoost, LightGBM) to estimate Remaining Useful Life (RUL) and trigger advance maintenance warnings.

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

**what We Are Predicting**

The goal is to predict the Remaining Useful Life (RUL) — how many operational cycles an engine has left before it fails.

**Engine Life Timeline:**

<img width="498" height="253" alt="image" src="https://github.com/user-attachments/assets/1b68b671-c80a-4570-a2ce-276082f06c68" />

**Phase 1 — Data Exploration**
Before building any model, the raw sensor data was explored to understand how engine health changes over time and which sensors show meaningful degradation patterns.

**Sensor Degradation Plot**
The plot below shows sensor readings for 4 different engines over their full lifetime. Each line represents one sensor tracked from the first cycle until engine failure.

![image alt](https://github.com/Sakshi290314/Predictive-Maintenance-of-Aircraft-Engines-Using-NASA-C-MAPSS Dataset/blob/851f2e246336fd2b25b18d396ed2f71b12e0cc81 / results/engine_degradation.png)

Shows how sensor readings change as engines approach failure. Clear degradation patterns visible.

**Phase 2 — Data Preprocessing**
Step 1 — Load Raw Data

Loaded train_FD001.txt with proper column names and removed empty columns caused by trailing spaces.

Step 2 — Calculate RUL

Created the target variable that the model will predict.

RUL = Max Cycle of Engine − Current Cycle

Engine 1 at Cycle 50 → RUL = 192 - 50 = 142 cycles left

Step 3 — Remove Useless Sensors

Sensors that never change were dropped since they add no value to the model.

Removed : sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19

Kept    : 14 sensors out of 21

Step 4 — Correlation Analysis

Checked how strongly each remaining sensor relates to RUL. The graph above shows sensors with strong positive or negative correlation — these are the most important features for predicting engine failure.

Step 5 — Normalize Features

All 14 sensors + 3 operational settings were scaled to 0–1 range using MinMaxScaler so no feature dominates due to its measurement unit.

**Models Overview (Simple Explanation)
Machine Learning Models (Traditional Approach)**
<img width="524" height="315" alt="image" src="https://github.com/user-attachments/assets/6fe97cfb-b3ea-4df1-b781-1dfafa92d9da" />

**Hyperparameter Tuning (Making Models Better)**

**Installation & Setup**

**Usage**

**Use Trained Model for Predictions**

**Results Summary
Performance Comparison**


**Project Structure**
Predictive-Maintenance-of-Aircraft-Engines-Using-NASA-C-MAPSS-Dataset/
│
├── 📁 data/                          # Dataset files
│   ├── train_FD001.txt               # Raw training data (100 engines)
│   ├── test_FD001.txt                # Raw test data (100 engines)
│   ├── RUL_FD001.txt                 # Ground truth RUL values
│   ├── train_processed.csv           # Cleaned & preprocessed data
│   └── readme.txt                    # Dataset documentation
│
├── 📁 models/                        # Saved trained models
│   ├── random_forest.pkl             # Random Forest model
│   ├── xgboost.pkl                   # XGBoost model
│   ├── lightgbm.pkl                  # LightGBM model
│   ├── scaler.pkl                    # Feature scaler
│   └── feature_columns.pkl           # Selected feature names
│
├── 📁 results/                       # Visualizations & evaluation results
│   ├── FINAL_COMPREHENSIVE_COMPARISON.png
│   ├── performance_improvement_chart.png
│   ├── sensor_correlations.png
│   └── other_visualizations.png
│
├── 📄 download_dataset.py            # Script to download NASA C-MAPSS data
├── 📄 01_data_exploration.py         # Exploratory Data Analysis (EDA)
├── 📄 02_data_preprocessing.py       # Data cleaning & feature engineering
├── 📄 03_ml_baseline.py              # Baseline model training
├── 📄 08_hyperparameter_tuning.py    # Model optimization
├── 📄 09_final_comparison.py         # Final model comparison & evaluation
│
├── 📄 requirements.txt               # Required Python libraries
└── 📄 README.md                      # Project documentation


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
