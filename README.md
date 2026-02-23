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

Checked how strongly each remaining sensor relates to RUL. 

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

<img width="543" height="446" alt="image" src="https://github.com/user-attachments/assets/7c3c19b5-d31f-41dd-9cc0-49407ae7cbb2" />

Process:
  Random Search → Try 20 different combinations
               → Pick the best performing one
               → Improves accuracy by 2-5%

<img width="543" height="541" alt="image" src="https://github.com/user-attachments/assets/d8df8127-5a6e-4868-9d9f-50cd05ef7fd4" />

Process:
  Random Search → Try 20 different combinations
               → Pick the best performing one
               → Improves accuracy by 2-5%

**Use Trained Model for Predictions**

<img width="729" height="569" alt="image" src="https://github.com/user-attachments/assets/fad1792e-bab3-4da6-b5e6-ff7a89aa6871" />

**Results Summary**

**Performance Comparison**

<img width="614" height="228" alt="image" src="https://github.com/user-attachments/assets/cebe5586-63ab-43db-9a7e-55eb0600cc72" />

**Visual Performance Comparison**

**Comprehensive Model Comparison:**

<img width="1692" height="958" alt="image" src="https://github.com/user-attachments/assets/eed86c87-9ecc-4ae8-bbcb-95617cf51739" />

**Performance Improvement Chart:**

<img width="1751" height="777" alt="image" src="https://github.com/user-attachments/assets/ff14fcb9-065a-4784-95fc-ac0b61f4cb94" />

Among the machine learning models tested, XGBoost achieved the best results in terms of accuracy and error reduction.

**Project Structure**

<img width="598" height="522" alt="image" src="https://github.com/user-attachments/assets/ae7ad53f-67b2-40a6-b6fb-262c6b277d29" />
<img width="610" height="503" alt="image" src="https://github.com/user-attachments/assets/98680d2c-1e64-47d4-86ce-5d02aed16465" />

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
