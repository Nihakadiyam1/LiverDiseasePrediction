# 🏥 Liver Disease Prediction App

This is a **Streamlit-based web application** that predicts **liver disease** based on patient details. It uses **Machine Learning (Gradient Boosting Classifier)** to analyze health parameters and predict whether a patient has liver disease or not.

---

## 🚀 Features

✅ Predicts **Liver Disease** using a trained ML model  
✅ Uses **Gradient Boosting Classifier** for prediction  
✅ **User-friendly Interface** built with **Streamlit**  
✅ **Real-time prediction** based on patient input  
✅ **Deployed on Hugging Face**

---

## 📁 Project Structure

LDD/ │── app.py # Main Streamlit app
│── countOfTar.py # Checks class distribution
│── test.py # Tests train-test split
│── Liver_disease_data.csv # Dataset
│── requirements.txt # Python dependencies
│── How to run.txt # Instructions
│── README.md # Project documentation

---

## 🛠 Installation & Setup

### 1️⃣ **Clone the Repository**

```sh
git clone https://github.com/yourusername/LiverDiseasePrediction.git
cd LiverDiseasePrediction

pip install -r requirements.txt --user

python -m streamlit run app.py

📊 Dataset
The dataset used is Liver_disease_data.csv
It contains patient records with various health parameters and their liver disease diagnosis.
🛠 Technologies Used
🔹 Python (pandas, numpy, scikit-learn)
🔹 Machine Learning (Gradient Boosting Classifier)
🔹 Streamlit (for UI)

🎯 How the Model Works
Step 1: Load patient data
Step 2: Preprocess input & apply Standard Scaling
Step 3: Pass the scaled input to the trained ML model
Step 4: Display prediction results

🚀 Live Demo
![image](https://github.com/user-attachments/assets/446ed179-6e76-4ecf-80e8-f7655c0c35a6)

output:
![image](https://github.com/user-attachments/assets/609372d1-9feb-4517-8bb0-4086aed5b285)




By Harika Kadiyam
from SVES
22a81a0622

```
