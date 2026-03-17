# 🏦 Bank Churn Prediction App

> MLOps Assignment 02 — FastAPI + Streamlit + RandomForest

---

## 📁 Project Structure

```
bank_churn_app/
│
├── Churn_Modelling.csv          ← Put dataset HERE (download from Kaggle)
├── train_model.py               ← STEP 1: Run this first
│
├── backend/
│   ├── model.pkl                ← Auto-created after running train_model.py
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       └── model.py
│
└── frontend/
    ├── app.py
    └── requirements.txt
```

---

## ⚙️ Setup & Run Order

### STEP 1 — Train the Model
```bash
cd bank_churn_app
pip install pandas scikit-learn
python train_model.py
```

### STEP 2 — Start Backend (Terminal 1)
```bash
cd bank_churn_app/backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API → http://127.0.0.1:8000
Docs → http://127.0.0.1:8000/docs

### STEP 3 — Start Frontend (Terminal 2)
```bash
cd bank_churn_app/frontend
pip install -r requirements.txt
streamlit run app.py
```
UI → http://localhost:8501

---

## 📬 Example Input & Output

Input:
```json
{
  "CreditScore": 650, "Geography": "France", "Gender": "Male",
  "Age": 40, "Tenure": 5, "Balance": 50000.0,
  "NumOfProducts": 2, "HasCrCard": 1,
  "IsActiveMember": 0, "EstimatedSalary": 75000.0
}
```

Output:
```json
{ "prediction": 1, "churn_probability": 0.64, "result": "Churn" }
```
