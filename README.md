# Forecasting Ambient Temperature with RNNs  
### A Comparison of Linear Models and Sequence Networks on the Jena Climate Dataset  
**Author:** Mintesnot Kassa  
**Course:** Machine Learning – Fordham University  
**Date:** Fall 2025  

---

## 📌 Project Overview
This project investigates short-horizon ambient temperature forecasting using real meteorological data from the **Jena Climate Dataset**. The goal is to evaluate how well **classical linear models** compare with **recurrent neural networks (RNNs)** such as **LSTM** and **GRU** when forecasting 1, 2, and 6 hours into the future.

The project includes:
- An end-to-end preprocessing and modeling pipeline  
- Exploratory data analysis (EDA) with multiple figures  
- Implementation of classical and deep learning forecasting models  
- Rigorous time-aware evaluation (no data leakage)  
- A full 6-page IEEE-style research report  

---

## 🧠 Problem Formulation
Given multivariate weather sequences sampled every **10 minutes**, the task is to predict **future ambient temperature** at multiple horizons:

- **1 hour ahead** (6 steps)  
- **2 hours ahead** (12 steps)  
- **6 hours ahead** (36 steps)

Using a sliding-window approach, each model receives a history window of past observations and must predict future temperature values. This problem is relevant for HVAC control, energy management, and short-term weather adaptation.

---

## 📂 Repository Structure

project/
│
├── project.ipynb # Main notebook with preprocessing, EDA, modeling, and evaluation
├── report.pdf # Final IEEE 2-column project report (6+ pages)
├── report.tex # LaTeX source for the report
│
├── figures/ # All figures used in the report
│ ├── fig_temp_time.png
│ ├── fig_temp_hist.png
│ ├── fig_corr_heatmap.png
│ ├── fig_daily_profile.png
│ ├── fig_error_by_hour.png
│ ├── fig_forecast_example.png
│ ├── fig_loss_curves_lstm.png
│ └── fig_rmse_bar.png
│
├── data/
│ └── jena_climate_2009_2016.csv # (If allowed — otherwise user downloads from Kaggle)
│
└── README.md


---

## 📊 Dataset Description
The **Jena Climate Dataset** contains weather measurements collected near the  
Max Planck Institute for Biogeochemistry (Germany) from **2009–2016** at 10-minute intervals.

Key features include:
- Temperature  
- Air pressure  
- Dew point  
- Relative humidity  
- Wind speed & gust  
- Water vapor content  
- Air density  

The target variable is **`T (degC)`** (ambient temperature).

---

## 📈 Models Implemented

### **Baseline Models**
- **Persistence Model** (last value carried forward)  
- **Linear Regression** on lag-window features  
- **Ridge Regression** with cross-validated penalty  

### **Deep Learning Models**
- **LSTM**  
- **GRU**  
Both trained on standardized sliding windows with:
- 32–64 hidden units  
- Dropout regularization  
- Adam optimizer  
- Early stopping  

---

## 🧪 Evaluation Strategy

### ✔ Time-Aware Splitting
The dataset is split chronologically to avoid temporal leakage:
- 60% training  
- 20% validation  
- 20% test  

### ✔ Metrics
- **RMSE** (Root Mean Squared Error)  
- **R² (Coefficient of Determination)**  
- Horizon-by-horizon comparison  

### ✔ Diagnostic Plots
The project includes visual evaluations such as:
- Daily temperature profile  
- Forecast vs. truth overlays  
- RMSE bar charts  
- Model architecture diagram  
- Error-by-hour-of-day analysis  

---

## 🔍 Key Findings
- **Persistence and linear regression perform strongly for 1-hour forecasting** due to smooth temperature dynamics.
- **Ridge regression improves stability** for longer windows with more features.
- **LSTM and GRU models outperform classical models at 6-hour horizons**, showing their ability to capture nonlinear temporal structure.
- **Forecast errors peak during sunrise and sunset**, when temperature changes fastest.

---

## ▶️ How to Run the Project

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
