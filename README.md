# Arch-Technology-Internship
This repository contains two interactive machine learning web applications developed using **Streamlit** and designed to run both **locally** and on **Google Colab with ngrok tunneling**.

## 📌 Projects Included


### 1. 📧 Email Spam Detector
A basic spam classifier that identifies whether an input email message is **Spam** or **Not Spam** using text classification.

#### 🔧 Features
- Input a full email or message
- Uses CountVectorizer + Multinomial Naive Bayes
- Instant spam classification

---

### 2. 🖍️ MNIST Digit Recognizer
An interactive web app that uses a Convolutional Neural Network (CNN) trained on the **MNIST dataset** to recognize handwritten digits (0-9) drawn by the user.

#### 🔧 Features
- Draw a digit on a canvas (live in browser)
- Preprocesses the input image
- Predicts using a trained CNN model
- Real-time classification (0–9)

---


## 🚀 Running the Apps

### ✅ Option 1: Run Locally

#### Step 1: Clone the repository
```bash
git clone https://github.com/RabailFiaz/Arch-Technology-Internship.git
cd Arch-Technology-Internship
```
#### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Run the app
```bash
# For Email Spam Detector
streamlit app.py
```
#### OR
#### Run on Google Colab
```bash
!pip install streamlit pyngrok
from pyngrok import ngrok

# Kill previous tunnels
ngrok.kill()

# Launch the app
!streamlit run Digit Recognition.ipynb &>/content/log.txt &

# Create public URL
public_url = ngrok.connect(port='8501')
print(f"🔗 App is live at: {public_url}")
```

#### 📁 Project Structure
📦 Arch Tech Internship/
├── Mnist Digit Recognition             
├── Email spam Detection         
└── README.md

#### 📊 Models Used
- MNIST Digit Recognizer: Keras CNN (trained on tensorflow.keras.datasets.mnist)
- Spam Detector: Scikit-learn's MultinomialNB with CountVectorizer

#### 🛠️ Requirements

streamlit
pyngrok
numpy
tensorflow
pandas
scikit-learn
Pillow


#### 📜 License
MIT License. Feel free to use, modify, and share!


#### 🙌 Acknowledgements

MNIST Dataset
Streamlit
Scikit-learn
Pyngrok


---

Let me know if you'd like:
- A sample `requirements.txt`
- Screenshots added to this README
- A GitHub repository name suggestion

I'm happy to help set up a complete GitHub push-ready structure.


