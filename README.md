# Credit-Card-Fraud-Detection-using-Logistic-Regression-and-XGboost
📌 Overview
This project implements a credit card fraud detection system using supervised machine learning techniques in R. The dataset is highly imbalanced, with fraudulent transactions forming a very small fraction of total observations. To address this, cost-sensitive learning and robust evaluation metrics are employed instead of naive accuracy-based approaches.

📊 Dataset
- Source: Public credit card transaction dataset
- Records: European cardholders’ transactions
- Features:
  - `Time`: Time elapsed since first transaction
  - `Amount`: Transaction amount
  - `V1–V28`: PCA-transformed anonymized features
  - `Class`: Target variable (0 = Non-Fraud, 1 = Fraud)

🔍 Exploratory Data Analysis
- Severe class imbalance identified
- Density plots used to compare transaction timing
- Histograms used to analyze transaction amounts
- Log-scaled plots for better visualization of skewed distributions

⚙️ Data Preprocessing
- Target variable converted to factor
- Time and Amount standardized
- Stratified train–test split (70–30)

🧠 Models Implemented

1️⃣ Logistic Regression (Class-Weighted)
- Baseline interpretable model
- Class imbalance handled using higher weights for fraud class
- Lower probability threshold used to improve fraud recall

2️⃣ XGBoost (Imbalance-Aware)
- Tree-based ensemble model
- Handles non-linear relationships and feature interactions
- Class imbalance handled using `scale_pos_weight`
- Optimized for ROC–AUC

📈 Model Evaluation
- Confusion Matrix
- ROC Curve
- AUC Score

Accuracy is not emphasized due to imbalance; ROC–AUC and recall are prioritized.

📉 Results
- Logistic Regression provides interpretability and stable baseline performance
- XGBoost achieves superior ROC–AUC and better fraud detection capability
- Combined ROC curves clearly show performance differences

🛠 Libraries Used
- `dplyr`
- `ggplot2`
- `caret`
- `pROC`
- `xgboost`
- `rpart`, `rpart.plot`

🧾 Conclusion
Cost-sensitive learning significantly improves fraud detection in imbalanced datasets. While Logistic Regression offers interpretability, XGBoost captures complex fraud patterns more effectively, making it better suited for real-world deployment.
