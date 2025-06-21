# 🛡️ Fall Risk Alert System

A smart fall detection alert system built with **Flask**, **Machine Learning (Random Forest)**, and **Python automation tools**. It detects potential fall risks based on simulated health sensor data (temperature, heartbeat, vision, and muscle angle), then **sends an email and plays an audio alert** if a fall risk is detected.

---

## ✨ Features

- 📊 Machine Learning model (Random Forest) trained on synthetic health data
- 📧 Sends automated **email alerts** using Gmail SMTP
- 🔊 Plays **audio warnings** for local alerting
- 🌐 Simple Flask web interface to simulate submission
- 🔐 Uses app-specific password for secure email sending (Gmail with 2FA)

---

## 🧰 Tech Stack

- Python 3.8+
- Flask
- scikit-learn
- pandas, numpy
- gTTS (Google Text-to-Speech)
- smtplib (Gmail)
- HTML (for form-based input)

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Eniyasre-AG/fall_detection.git
cd fall_detection/SIh
