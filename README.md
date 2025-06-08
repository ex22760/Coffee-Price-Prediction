# Real World Arabica Coffee Price Prediction

Forecasting climate-sensitive variables is a critical task across domains like agriculture, finance, energy, and disaster planning. One of the most influential climate patterns is the El Niño–Southern Oscillation (ENSO), which affects global weather and economic outcomes. ENSO phases - El Niño, La Niña, and Neutral - are commonly captured through indices such as the Oceanic Niño Index (ONI).

In this notebook, I aim to forecast a time series target variable (e.g. a climate, environmental, or economic indicator) using ENSO-related features. I explore multiple modelling approaches to understand the impact of ENSO signals and optimise predictive performance:

- Classical time series models (e.g. SARIMAX, VAR)

- Supervised machine learning models (XGBoost, Random Forest, Gradient Boosting)

- Deep learning models (LSTM)

- Model ensembling (e.g. Stacked Ensemble)

- Interpretability via SHAP analysis

I also introduce ENSO-aware feature engineering and rolling forecasts to simulate real-world deployment. Models are compared using standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE).

This project demonstrates how ENSO signals can be leveraged for high-accuracy forecasting using modern data science techniques.

---

## Tech Stack

- **Languages:** Python

- **Libraries:**
  - **Data Manipulation & Analysis:** `pandas`, `numpy`
  - **Visualisation:** `matplotlib`, `seaborn`
  - **Time Series Modeling:** `statsmodels` (SARIMAX, VAR, decomposition, stationarity tests)
  - **Machine Learning:** `scikit-learn` (Random Forest, Gradient Boosting, RFE, RFECV, Stacking), `xgboost`
  - **Deep Learning:** `tensorflow`, `keras` (LSTM)
  - **Model Explainability:** `shap`
  - **Utilities:** `datetime`, `random`, `os`, `re`, `warnings`

- **Tools:**
  - **Development Environment:** Jupyter Notebook / JupyterLab
  - **Version Control:** Git & GitHub
  - **Model Optimisation:** `GridSearchCV`, `TimeSeriesSplit`, `Pipeline`
  - **Visualisation & Debugging:** SHAP plots



---

## Project Highlights

## Key Insights

- **Best Individual Model:**  
  The *Simple + RFE selected engineered features* model remains the top-performing standalone model with the lowest MAE (0.0978) and MAPE (6.04%), making it the most accurate and reliable for direct deployment.

- **LSTM + XGBoost Ensemble:**  
  The *Ensemble model* that combines LSTM and XGBoost predictions achieves competitive performance (MAE: 0.1119, MAPE: 6.83%). This hybrid approach captures both temporal dynamics and engineered feature interactions, offering robustness and model diversity.

- **Tree-Based Models:**  
  Both *Random Forest* and *XGBoost* demonstrate strong performance with low errors, confirming their effectiveness in capturing complex relationships in structured feature sets.

- **LSTM Alone:**  
  Despite enhancements like the `is_stable_month` feature, the LSTM model still performs worse than classical ML models, indicating limited value from deep learning in this particular use case.

- **Traditional Time Series Models:**  
  Classical methods such as *SARIMAX* and *VAR*, even after differencing and transformations to ensure stationarity, performed poorly relative to ML models. This reinforces the advantage of flexible machine learning models over rigid statistical assumptions in capturing the complexity of this dataset.

---

## Final Recommendation

- **Best Option:** Use the *Simple + RFE selected engineered features* model for its top accuracy and simplicity.
- **Alternative (Hybrid):** Consider deploying the *LSTM + XGBoost Ensemble* for situations requiring robustness across temporal patterns and feature-based variance.
  
---

## Dashboard

To make the ensemble model results easily interpretable and accessible, I developed a responsive Streamlit dashboard. It allows users to upload predictions, view model performance metrics (MAE, MSE, MAPE), and interactively compare actual vs. forecasted values in an intuitive plot.

![image](https://github.com/user-attachments/assets/17423af5-07b8-4be2-928f-2fe7163337a0)

---

## Workflow 

This notebook walks through the following key steps:

### 1. Data Overview + Exploratory Data Analysis (EDA) 
- Dataset Merging and Structure
- Summary Stats and Missing Data
- Time Series Plots and Visual Comparisons
- Stationarity and Decomposition insights
- Volatility (Rolling STD) analysis

### 2. Feature Engineering
- Correlation Analysis with Lag Features
- Seasonal and Rolling Average Features
- Categorical ENSO Phases
- Interaction Features
- ENSO Timing Feature over Time
- Correlation of Engineered Features to Price

### Modelling 

- Traditional Models
  - SARIMA (seasonality only)
  - Seasonal SARIMAX (log-diff + exog + seasonality)
  - VAR with Differencing
    
- Machine Learning Models
  - Gradient Boosting
    - Engineered features only (correlation-based)
    - Simple model (no engineered features)
    - Simple + RFE selected engineered features
    - Simple + RFECV selected features
  - XGBoost
    - Standard with GridSearch
    - With Rolling Forecast
  -RandomFOrest
    - Standard with GridSearch
    - With Regularisation
  - LSTM
    - Standard
    - With 'is_stable_month' flag
      
- Stacked Ensemble Model (LSTM + XGBoost)

---

## Results


| Model                               | MAE    | MAPE   | MSE    |
|-------------------------------------|--------|--------|--------|
| Updated LSTM (is_stable_month)      | 0.1245 | 7.65%  | 0.0337 |
| Random Forest (Regularised)         | 0.1028 | 6.64%  | 0.0218 |
| XGBoost with Rolling Forecast       | 0.1022 | 6.24%  | 0.0216 |
| VAR (with differencing)             | 0.6978 | 0.5156 | 28.11% |
| Simple + RFECV selected features    | 0.0937 | 5.90%  | 0.0188 |
| **LSTM + XGBoost Ensemble**         | 0.1097 | 6.70%  | 0.0218 |


---

## How to Run 

1. Clone the repository:

```bash
git clone https://github.com/ex22760/Coffee-Price-Prediction.git
cd Coffee-Price-Prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the notebook:

```bash
jupyter notebook arabica-coffee-price-prediction.ipynb
```

---

## Folder Structure

```bash

├── data/                                  # Raw datasets extracted from online
├── coffee-price-prediction.ipynb          # Main notebook with code and analysis
├── requirements.txt
├── ensemble_dashboard.py                  # Code for streamlit dashboard (input data for dashboard in data/streamlit_input.csv)
├── README.md
├── LICENSE
```

---

## Acknowledgements

Developed initially as part of the module 'Mathematical Data Modelling 3' at the University Of Bristol: Engineering Mathematics course. 
Expanded from a single XGBoost model I created to a complete end to end pipeline repository including another personally researched dataset to improve model accuracy. 

---

## Contact

**Sujin Subanthran**

[LinkedIn Profile](https://www.linkedin.com/in/sujin-subanthran-b44512226/)
