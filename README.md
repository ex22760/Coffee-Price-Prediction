# Real World Arabica Coffee Price Prediction

Forecasting climate-sensitive variables is a critical task across domains like agriculture, finance, energy, and disaster planning. One of the most influential climate patterns is the El Niño–Southern Oscillation (ENSO), which affects global weather and economic outcomes. ENSO phases — El Niño, La Niña, and Neutral — are commonly captured through indices such as the Oceanic Niño Index (ONI).

In this notebook, I aim to forecast a time series target variable (e.g. a climate, environmental, or economic indicator) using ENSO-related features. I explore multiple modelling approaches to understand the impact of ENSO signals and optimise predictive performance:

- Classical time series models (e.g. SARIMAX, VAR)

- Supervised machine learning models (XGBoost, Random Forest, Gradient Boosting)

- Deep learning models (LSTM, Transformer-based)

- Model ensembling (e.g. Stacked Regressors)

- Interpretability via SHAP analysis

I also introduce ENSO-aware feature engineering and rolling forecasts to simulate real-world deployment. Models are compared using standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE).

This project demonstrates how ENSO signals can be leveraged for high-accuracy forecasting using modern data science techniques.

---

## Tech Stack

- **Languages:** Python

- **Libraries:**
  - **Data Manipulation & Analysis:** `pandas`, `numpy`
  - **Visualization:** `matplotlib`, `seaborn`
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

## Workflow xxxxxxxxxxxxxxxxxxxx

This notebook walks through the following key steps:

### 1. Exploratory Data Analysis (EDA)
- Survival rate by gender and age
- Distribution analysis of age and fares

### 2. Feature Engineering
- Extracted **honorifics** from passenger names
- Grouped **cabins** by deck
- Calculated **family size** and identified **single mothers**
- Removed redundant/collinear features (e.g. high correlation between `SibSp` and `FamilySize`)

### 3. Imputation Techniques Explored
- Mean/Median/Mode imputation (with group-based logic)
- Random imputation (performed best)
  
  ![image](https://github.com/user-attachments/assets/11f1ea5d-2d57-4a9a-a288-183a49678dbf)

  
  ![image](https://github.com/user-attachments/assets/660a508b-0502-4d61-9a5f-6e6981d346a9)


- Stratified random sampling
- Advanced methods: **MICE**, **MissForest**, **KNN**

### 4. Feature Engineering & Preprocessing
- Capped outliers
- Log, Box-Cox, and Yeo-Johnson transformations

  
  ![image](https://github.com/user-attachments/assets/1137d0a4-28b9-47ad-97e9-282f57584fe5)

- Encoded rare labels in categorical columns
- Calculated permutation-based feature importances:

  
![image](https://github.com/user-attachments/assets/a30ccf30-86cf-476d-955b-74f5e1c56279)



### 5. Modelling and Grid Search


| Pipeline                         | Train Accuracy | Test Accuracy |
|---------------------------------|----------------|---------------|
| Random Forest Classification     | 0.995          | 0.802         |
| Gradient Boosting Classification | 0.915          | 0.802         |
| Logistic Regression              | 0.839          | 0.821         |
| **Random Forest with Grid Search** | **0.851**    | **0.825**     |
| Gradient Boosting with Grid Search | 0.909        | 0.795         |
| Logistic Regression with Grid Search | 0.833       | 0.810         |


  - Random Forest with Grid search performed best with best parameters: `max_depth=5`, `min_samples_split=10`, `n_estimators=30`


---

## Final Classification Report

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 0       | 0.81      | 0.91   | 0.86     | 157     |
| 1       | 0.85      | 0.70   | 0.77     | 111     |
| **Accuracy** |           |        | **0.82**   | 268     |
| **Macro Avg** | 0.83      | 0.81   | 0.81     | 268     |
| **Weighted Avg** | 0.83  | 0.82   | 0.82     | 268     |


---

## How to Run 

1. Clone the repository:

```bash
git clone https://github.com/ex22760/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the notebook:

```bash
jupyter notebook titanic_survival_analysis.ipynb
```

---

## Folder Structure

```bash

├── data/                   # Raw datasets used for the competition
├── titanic_notebook.ipynb  # Main notebook with code and analysis
├── submission_titanic.csv  # Final prediction file submitted to Kaggle
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
```

---

## Acknowledgements

Developed as part of the Titanic: Machine Learning from Disaster competition.

![image](https://github.com/user-attachments/assets/609478e2-b15a-4588-8f61-cdd6716bdf9e)

---

## Contact

**Sujin Subanthran**

[LinkedIn Profile](https://www.linkedin.com/in/sujin-subanthran-b44512226/)
