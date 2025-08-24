# 🧠 Machine Learning Internship Projects  

This repository contains **four machine learning projects** developed during the internship.  
They are divided into two folders:  

- 📂 **Arch-Technology-Internship** → Interactive web apps (Streamlit + Colab ngrok)  
- 📂 **Month2** → Core ML models (Housing Price Prediction + Iris Flower Classification)  

---

## 📌 Projects Included  

### 📂 Arch-Technology-Internship  

#### 1. 📧 Email Spam Detector  
A basic spam classifier that identifies whether an input email message is **Spam** or **Not Spam** using text classification.  

**Features**  
- Input a full email or message  
- Uses **CountVectorizer + Multinomial Naive Bayes**  
- Instant spam classification  

---

#### 2. 🖍️ MNIST Digit Recognizer  
An interactive web app that uses a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** to recognize handwritten digits (0–9) drawn by the user.  

**Features**  
- Draw a digit on a canvas (live in browser)  
- Preprocesses the input image  
- Predicts using a trained CNN model  
- Real-time classification (0–9)  

---

### 📂 Month2  

#### 3. 🏠 California Housing Price Prediction  
A regression model that predicts **median house prices** in California based on features like **income, number of rooms, population, and location**.  

**Models Used**  
- Linear Regression  
- Random Forest Regressor  

**Evaluation Metrics**  
- RMSE (Root Mean Squared Error)  
- R² Score  

**Key Insights**  
- Median income is the strongest predictor.  
- Random Forest outperforms Linear Regression with better accuracy.  

---

#### 4. 🌸 Iris Flower Classification  
A classification model to predict **iris species** (Setosa, Versicolor, Virginica) using **sepal length, sepal width, petal length, and petal width**.  

**Models Used**  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  

**Evaluation Metrics**  
- Accuracy  
- Confusion Matrix  
- Classification Report  

**Key Insights**  
- KNN performed best (~98% accuracy).  
- Setosa is always classified correctly; most misclassifications occur between Versicolor and Virginica.  

---

## 🚀 Running the Apps  

### ✅ Option 1: Run Locally  
```bash
git clone https://github.com/RabailFiaz/Arch-Technology-Internship.git
cd Arch-Technology-Internship
pip install -r requirements.txt
```

## Run Streamlit Apps (Arch Tech Internship)
```bash
# For Email Spam Detector
cd "Arch-Technology-Internship/Email spam Detection"
streamlit run app.py

# For MNIST Digit Recognizer
cd "Arch-Technology-Internship/Mnist Digit Recognition"
streamlit run app.py
```
## Run Month2
```bash
# Housing Price Prediction
cd Month2
python task3.ipynb

# Iris Classification
cd Month2
python task4.ipynb

```

## Option 2: Run on Google Colab (Arch Tech Internship apps)
```bash
!pip install streamlit pyngrok
from pyngrok import ngrok

# Kill previous tunnels
ngrok.kill()

# Launch the app
!streamlit run Digit_Recognition.ipynb &>/content/log.txt &

# Create public URL
public_url = ngrok.connect(port='8501')
print(f"🔗 App is live at: {public_url}")
```
## 📁 Project Structure
📦 Arch-Technology-Internships/
├── Arch-Technology-Internship/
│   ├── Mnist Digit Recognition
│   ├── Email spam Detection
│   └── README.md
│
├── Month2/
│   ├── task3.ipynb
│   ├── task4.ipynb
│   └── README.md



## 📜 License

MIT License – feel free to use, modify, and share!

## 🙌 Acknowledgements
MNIST Dataset
Iris Dataset (UCI Machine Learning Repository)
Scikit-learn
TensorFlow / Keras
Streamlit
Pyngrok


