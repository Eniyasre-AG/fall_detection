import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, render_template
import smtplib
from email.message import EmailMessage
from gtts import gTTS
import os

app = Flask(__name__)

# Email configuration for Gmail
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465
SMTP_USER = 'roomies1224@gmail.com'  # Replace with your email address
SMTP_PASSWORD = 'jgxc szzw iedy gwfe'  # Replace with your app-specific password (2FA)

# File path for the model
MODEL_FILE_PATH = 'fall_alert_model.pkl'

# Function to train and save the model if not already saved
def train_and_save_model():
    np.random.seed(42)
    n_samples = 1000
    temperature_normal = 37.0
    heartbeat_normal = 75
    light_vision_normal = 500
    angle_normal = 0

    temperature = np.random.normal(temperature_normal, 1.5, n_samples)
    heartbeat = np.random.normal(heartbeat_normal, 10, n_samples)
    light_vision = np.random.normal(light_vision_normal, 100, n_samples)
    muscle_angle = np.random.normal(angle_normal, 5, n_samples)

    alert = (temperature > 38) | (heartbeat > 85) | (light_vision < 400) | (np.abs(muscle_angle) > 10)
    alert = alert.astype(int)

    data = pd.DataFrame({
        'temperature': temperature,
        'heartbeat': heartbeat,
        'light_vision': light_vision,
        'muscle_angle': muscle_angle,
        'alert': alert
    })

    X = data.drop('alert', axis=1)
    y = data['alert']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model to a file
    with open(MODEL_FILE_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)

    print("Model has been saved to 'fall_alert_model.pkl'.")

# Train the model if the model file does not exist
if not os.path.exists(MODEL_FILE_PATH):
    train_and_save_model()

# Load model
with open(MODEL_FILE_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to send an email alert
def send_email_alert(subject, body, recipient):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['To'] = recipient
    msg['From'] = SMTP_USER

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
            print("Email sent successfully.")
    except smtplib.SMTPException as e:
        print(f"SMTP error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Function to generate and play audio alert
def send_audio_alert(message="Fall risk detected. Immediate attention required."):
    tts = gTTS(text=message, lang='en')
    tts.save("fall_alert.mp3")
    os.system("start fall_alert.mp3")  # For Windows
    # os.system("open fall_alert.mp3")  # For macOS
    # os.system("xdg-open fall_alert.mp3")  # For Linux

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    recipient_email = request.form.get('email')

    # Simulated new data for prediction
    new_data = pd.DataFrame({
        'temperature': [39.0],
        'heartbeat': [90],
        'light_vision': [350],
        'muscle_angle': [12]
    })

    # Predict if an alert is needed
    new_prediction = model.predict(new_data)

    if new_prediction[0] == 1:
        send_email_alert(
            subject="Fall Risk Alert",
            body="Alert: A fall risk has been detected based on the latest data.",
            recipient=recipient_email
        )
        send_audio_alert("Alert! A fall risk has been detected. Please take immediate action.")

    return f"Alert sent to {recipient_email}."

if __name__ == '__main__':
    app.run(debug=True)