# prediction-ML-model
Built a machine learning-based liver disease prediction system using 5500+ patient records, implementing multiple classification algorithms and data preprocessing techniques to achieve accurate early disease detection and healthcare risk analysis.



# 🩺 Liver Disease Prediction using Machine Learning

An advanced end-to-end Machine Learning project for predicting multiple liver disease conditions using clinical, biochemical, and lifestyle patient data.

This project combines:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Multi-model machine learning
- Performance benchmarking
- Feature importance analysis
- Real-time patient prediction
- Professional healthcare visualizations

Built using Python, Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.

---

# 🚀 Project Overview

This system predicts liver disease categories from patient medical records and lifestyle indicators using supervised machine learning models.

The project was developed using a dataset containing **5500+ patient records** and **33+ healthcare-related features**.

The pipeline includes:

✔ Data Cleaning  
✔ Missing Value Handling  
✔ Feature Engineering  
✔ Visualization Dashboard  
✔ Multiple ML Models  
✔ Model Evaluation  
✔ Feature Importance Ranking  
✔ Real-Time Prediction System  

---

# 🧠 Disease Classes Predicted

The model predicts the following categories:

- Healthy Liver
- Fatty Liver Disease (NAFLD)
- Alcoholic Liver Disease
- General Liver Disease Severity
- Liver Cirrhosis Risk

---

# 📊 Dataset Information

| Attribute | Details |
|---|---|
| Dataset Size | 5500+ Patients |
| Features | 33+ Clinical & Lifestyle Features |
| Problem Type | Multi-Class Classification |
| Domain | Healthcare AI / Medical Analytics |

---

# ⚙️ Technologies Used

## Programming Language
- Python 3.x

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 🏗️ Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

# 🔍 Key Features

## ✅ Advanced Data Preprocessing
- Missing value handling
- Median imputation
- Label encoding
- Ordinal encoding
- Feature scaling

## ✅ Exploratory Data Analysis
Automatically generates professional plots for:
- Disease distribution
- Liver enzyme analysis
- Lifestyle factors
- BMI distribution
- Correlation heatmaps
- Gender & age analysis

## ✅ Model Evaluation
Compares models using:
- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation

## ✅ Feature Importance Analysis
Identifies the most important clinical indicators affecting liver disease prediction.

## ✅ Real-Time Prediction Mode
Allows users to manually enter patient data and get disease predictions instantly.

---

# 📈 Generated Visualizations

The project automatically creates multiple healthcare analytics dashboards inside the `plots/` folder.

## Generated Outputs
```text

plots/
│
├── 01_class_distribution.png
├── 02_enzymes_by_class.png
├── 03_lifestyle_factors.png
├── 04_bmi_distribution.png
├── 05_correlation_heatmap.png
├── 06_age_gender.png
├── 07_model_comparison.png
├── 08_all_metrics.png
├── 09_confusion_matrices.png
├── 10_feature_importance.png
└── 11_dashboard.png
```

---

# 📂 Project Structure

```text
liver-disease-prediction/
│
├── liver_project.py
├── Training_Liver_Disease_Dataset.csv
├── plots/
├── requirements.txt
└── README.md
```

---

# 🖥️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/liver-disease-prediction.git
cd liver-disease-prediction
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Project

```bash
python liver_project.py
```

---

# 📦 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

# 🧪 Evaluation Metrics

Models are evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Cross Validation Score
- Confusion Matrix
- Classification Report

---

# 🧠 Important Features Used

## Clinical Features
- ALT
- AST
- Bilirubin
- Albumin
- Platelets
- GGT
- INR
- Triglycerides

## Lifestyle Features
- Alcohol Consumption
- Smoking Status
- Physical Activity
- Diet Quality

## Patient Information
- Age
- Gender
- BMI
- Waist Circumference
- Obesity Class

---

# 🖥️ Real-Time Prediction Example

```text
Gender: Male
BMI: 29
ALT: 68
AST: 71
Alcohol Consumption: Moderate
Smoking Status: Current
```

### Output

```text
Predicted Liver Condition:
Fatty Liver Disease (NAFLD)
```

---

# 📊 Best Performing Model

The system automatically selects the best-performing model based on:
- Highest F1 Score
- Accuracy
- Cross-validation performance

Random Forest generally performs best due to its strong ensemble learning capability and robustness.

---

# 🔬 Project Highlights

✔ End-to-End ML Pipeline  
✔ Healthcare Analytics Dashboard  
✔ Multi-Class Classification  
✔ Professional Data Visualization  
✔ Real-Time Prediction System  
✔ Production-Style ML Workflow  

---

# 🚀 Future Improvements

Possible future enhancements:

- Deep Learning Integration
- Flask / Django Web App
- Streamlit Dashboard
- REST API Deployment
- Explainable AI (SHAP/LIME)
- Docker Support
- Cloud Deployment

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed as an advanced Machine Learning and Healthcare Analytics project focused on liver disease prediction and biomedical data intelligence.

---

# ⭐ GitHub Tips

## Recommended Files to Upload
✅ README.md  
✅ requirements.txt  
✅ plots/ screenshots  
✅ clean source code  

## Avoid Uploading
❌ .env files  
❌ virtual environments (`venv/`)  
❌ cache folders  
❌ personal credentials  

---

# 📌 Skills Demonstrated

- Machine Learning
- Data Science
- Healthcare AI
- Feature Engineering
- Data Visualization
- Model Evaluation
- Python Development
- Statistical Analysis
- Biomedical Analytics

---
