# Real World Arabica Coffee Price Prediction

This project delivers a robust time series forecasting pipeline combining macroeconomic indicators and ENSO climate signals to predict Arabica coffee prices. Leveraging classical, machine learning, and deep learning models, I build an ensemble to enhance accuracy and deploy an interactive Streamlit dashboard for real-time forecast visualisation and performance monitoring.

---

## Background / Motivation

Forecasting climate-sensitive variables is critical across sectors like agriculture, finance, energy, and disaster risk management. The El Niño–Southern Oscillation (ENSO) — comprising El Niño, La Niña, and Neutral phases — is a key climate pattern influencing global weather and economic dynamics. These phases are commonly tracked via indices such as the Oceanic Niño Index (ONI).

In this project, I develop a robust time series forecasting pipeline leveraging macroeconomic indicators and ENSO-related features to predict Arabica coffee prices (USD/kg). I implement and benchmark multiple modelling approaches, including:

- Classical time series models (SARIMAX, VAR)

- Supervised machine learning models (XGBoost, Random Forest, Gradient Boosting)

- Deep learning models (LSTM)

- Ensemble techniques (stacked models) to improve forecast accuracy

Additionally, I incorporate ENSO-aware feature engineering and rolling forecast validation to better simulate deployment scenarios. To facilitate analysis and communication of results, I built an interactive Streamlit dashboard showcasing model predictions versus actuals, alongside key metrics (MAE, MAPE, MSE), enabling efficient exploration of forecast performance over time.

---

## Tech Stack

- **Languages:** Python

- **Libraries:**
  - **Data Manipulation & Analysis:** `pandas`, `numpy`
  - **Visualisation:** `matplotlib`, `seaborn`
  - **Time Series Modeling:** `statsmodels` (SARIMAX, VAR, decomposition, stationarity tests)
  - **Machine Learning:** `scikit-learn` (Random Forest, Gradient Boosting, RFE, RFECV, Stacking), `xgboost`
  - **Deep Learning:** `tensorflow`, `keras` (LSTM)
  - **Utilities:** `datetime`, `random`, `os`, `re`, `warnings`

- **Tools:**
  - **Development Environment:** Jupyter Notebook / JupyterLab
  - **Version Control:** Git & GitHub
  - **Model Optimisation:** `GridSearchCV`, `TimeSeriesSplit`, `Pipeline`
  - **Interactive Dashboarding** - `Streamlit` (for building a web app to visualise and compare model forecasts)

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

The notebook walks through the following key steps:

### 1. Data Overview + Exploratory Data Analysis (EDA) 
- Dataset Merging and Structure
- Summary Stats and Missing Data
- Time Series Plots and Visual Comparisons
  ![image](https://github.com/user-attachments/assets/e12d3784-4c3b-490d-b9cc-7d399a4740cf)
- Stationarity and Decomposition insights
- Volatility (Rolling STD) analysis
  ![image](https://github.com/user-attachments/assets/64f8ea5f-bdd5-4f14-ae42-0a5130c6ac35)

### 2. Feature Engineering
- Correlation Analysis with Lag Features
- Seasonal and Rolling Average Features
- Categorical ENSO Phases
- Interaction Features
- ENSO Timing Feature over Time
- Correlation of Engineered Features to Price
![image](https://github.com/user-attachments/assets/badec2bb-0979-4286-9368-a6a35d6f286a)


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

  -RandomForest
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

4: Run the Streamlit Dashboard:

```bash
streamlit run ensemble_dashboard.py
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
