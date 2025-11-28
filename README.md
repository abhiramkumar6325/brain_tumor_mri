# brain_tumor_mri


# TumorPredict – Federated Learning Brain Tumor Detection App

A smart medical imaging system powered by AI for **Brain MRI Tumor Classification** using both **online (server-based)** and **offline (on-device TFLite)** predictions.
The system also includes a **Doctor–Patient dashboard**, **chat system**, **report review**, and **Federated Learning weight uploads**.

---

## 🚀 Overview

TumorPredict is a cross-platform medical app designed for:

* Brain MRI classification (Tumor / No Tumor or multi-class)
* Doctor review, approve/reject, and add notes
* Patient–Doctor chat messaging
* Fetching prediction history and reports
* Federated Learning model weight upload
* Admin/Doctor-side aggregation
* Online prediction through Flask backend
* Offline prediction using TFLite on device

The app uses:

### 🟦 Frontend

**Flutter + Dart**

### 🟩 Backend

**Flask (Python)** — File-based JSON storage
(No MySQL)

### 🔶 AI / ML

**TensorFlow + TFLite**
Supports Federated Learning simulation

---

# 🧠 Features

### ✔️ AI-Based MRI Brain Tumor Classification

* Upload MRI
* Predict via online Flask model (demo model)
* Offline TFLite inference (coming soon)
* Saves predictions as structured JSON reports

---

### ✔️ Doctor Dashboard

* View all reports
* Pending reviews
* Approved and rejected reports
* Add doctor notes
* Approve / Reject each MRI report

---

### ✔️ Patient Dashboard

* View latest MRI result
* Prediction history
* Health tips
* My doctor page
* Chat with doctor
* Emergency help

---

### ✔️ Doctor–Patient Chat System

* WhatsApp-style chat
* Real-time polling every 3 seconds
* Message storage in server JSON file

---

### ✔️ Federated Learning Support

* Upload client weights (`.npz`)
* View global meta info
* Trigger server-side aggregation (stub)

---

### ✔️ Secure User Account System

* Register / Login
* OTP Forgot Password
* Role selection (Doctor or Patient)
* Stored using shared preferences on device

---

# 📱 App Flow

### Patient Side

1. Splash → Role Selection → Login
2. Dashboard
3. Upload MRI → Predict
4. View Results
5. Chat with doctor
6. Notification center

### Doctor Side

1. Splash → Role Selection → Login
2. Doctor Dashboard
3. View **Pending / All / Approved / Rejected** reports
4. Open report → Write notes → Approve/Reject
5. Federated Learning Tools
6. Chat with patient

---

# 🗂️ Project Structure (Flutter)

```
lib/
│── main.dart
│── api_service.dart
│── user_prefs.dart
│
│── splash_screen.dart
│── role_selection_screen.dart
│── login_screen.dart
│── register_screen.dart
│
│── chat/
│     ├── chat_screen.dart
│     ├── chat_service.dart
│     └── chat_message_model.dart
│
│── patient/
│     ├── dashboard_screen.dart
│     ├── health_tips_screen.dart
│     ├── upload_history_screen.dart
│     ├── mri_prediction_screen.dart
│     ├── patient_welcome_screen.dart
│     └── emergency_screen.dart
│
│── doctor/
      ├── doctor_dashboard.dart
      ├── doctor_patient_list.dart
      ├── doctor_report_review.dart
      ├── doctor_fl_tools.dart
      ├── doctor_navigation.dart
      └── doctor_profile.dart

assets/
│── images/
│      ├── background.jpg
│      ├── logo.png
│      ├── doctor.png
│      └── patient.png
│── models/
       └── brain_model.tflite
```

---

# 🖥️ Backend Details

### ✔️ Backend: Flask

Stored at:

```
backend/
│── server.py
│── train_brain_tumor.py
│── chat_messages.json
│── reports/
│── uploads/
│── fl_server/storage/users.json
```

Endpoints include:

| Endpoint                | Purpose                   |
| ----------------------- | ------------------------- |
| `/login`                | Login user                |
| `/register`             | Register user             |
| `/predict`              | Predict MRI + save report |
| `/list_reports`         | Fetch reports             |
| `/doctor_update_report` | Approve / Reject report   |
| `/send_message`         | Send chat message         |
| `/get_messages`         | Get chat history          |
| `/upload_weights`       | Upload FL weights         |
| `/trigger_aggregation`  | Start FL aggregation      |

---

# 🤖 AI/ML: Federated + Central Model

### Online Model (Flask)

* A lightweight demo prediction model
* Can be replaced with trained `.h5` model
* Returns:

```json
{
  "label": "No Tumor",
  "confidence": 0.93,
  "model_version": "v1"
}
```

### Offline Model (Device)

* Add `model.tflite` to:

```
assets/models/
```

### Federated Learning

* Clients upload `.npz` weights
* Server stores in `fl_server/storage`
* `trigger_aggregation` merges (demo only)

---

# 🛠️ Requirements

### Frontend

* Flutter SDK 3+
* Android Studio
* Dart
* TFLite Flutter plugin

### Backend

* Python 3.10 (Render compatible)
* Flask
* Flask-CORS
* Gunicorn
* Numpy

### ML

* TensorFlow 2.x
* TFLite Converter

---

# 📬 Support

If you need:

* Backend deployment
* Replacing demo model with real model
* Adding offline TFLite inference
* Improving UI
* Fixing chat system
  Just ask anytime.

