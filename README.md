Perfect! Here's your **updated README with emojis** to make it more friendly and visually appealing:

---

# Diabetes Prediction ML App 🩺🤖

A **Streamlit web app** to predict the likelihood of diabetes using health parameters and a trained **Random Forest Classifier** model. 🌟

---

## Features ✨

* Predicts diabetes risk from health parameters:
  🟢 Pregnancies
  🟢 Glucose
  🟢 Blood Pressure
  🟢 Skin Thickness
  🟢 Insulin
  🟢 BMI
  🟢 Diabetes Pedigree Function
  🟢 Age
* Uses **Random Forest Classifier**  for prediction
* Interactive **Streamlit UI** 
* Works locally and on **Streamlit Cloud** 

---

## Folder Structure 📂

```
diabetes-prediction-ml/
│
├─ data/
│   └─ diabetes.csv
├─ model/
│   ├─ diabetes_model.joblib
│   └─ scaler.joblib
├─ src/
│   └─ streamlit_app.py
├─ train.py
├─ requirements.txt
└─ README.md
```

---

## Installation ⚙️

1. Clone the repository:

```bash
git clone https://github.com/Kalaiyarasi27/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

2. Create & activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run src/streamlit_app.py
```

---

## Notes 

* Ensure the **`model/` folder** with `diabetes_model.joblib` and `scaler.joblib` is present. ✅
* Deployed version: [Streamlit App](https://diabetes-prediction-mlgit-e7kvfp4pzyupiu6rwnd2xu.streamlit.app/) 🌐

---

## Author 👩‍💻

**Kalaiyarasi N**
GitHub: [https://github.com/Kalaiyarasi27](https://github.com/Kalaiyarasi27)

---


