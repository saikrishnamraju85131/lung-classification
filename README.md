# 🫁 Lung Disease Classification System

A machine learning-powered web application for classifying lung diseases from audio recordings. This system uses XGBoost to analyze respiratory sounds and detect various pulmonary conditions including Asthma, COPD, Pneumonia, and Bronchial issues.

## 📋 Overview

This healthcare application provides doctors with an intelligent diagnostic tool to analyze patient respiratory sounds and classify potential lung conditions. The system combines advanced audio feature extraction with machine learning to deliver accurate, confidence-scored predictions.

### Supported Diagnoses
- **Bronchial** conditions
- **Asthma**
- **COPD** (Chronic Obstructive Pulmonary Disease)
- **Pneumonia**
- **Healthy** (no detected abnormalities)

## ✨ Key Features

### Patient Management
- **Doctor Authentication**: Secure login/signup system with password hashing
- **Patient Registration**: Comprehensive patient profiles including demographics and photos
- **Patient Search**: Quick search functionality by name or phone number
- **Patient History**: Track all recordings and diagnoses over time

### AI-Powered Diagnosis
- **Audio Analysis**: Upload respiratory sound recordings (WAV format)
- **Feature Extraction**: Advanced audio processing using librosa
  - Zero-crossing rate
  - Chroma STFT
  - MFCC (Mel-frequency cepstral coefficients)
  - RMS energy
  - Mel spectrogram
- **XGBoost Classification**: Pre-trained model for accurate disease detection
- **Confidence Scores**: Percentage-based confidence metrics for each diagnosis

### Data Management
- **SQLite Database**: Lightweight database for patient records
- **File Storage**: Organized storage for patient photos and audio recordings
- **Diagnosis History**: Complete audit trail of all classifications

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework for Python
- **SQLAlchemy**: ORM for database operations
- **XGBoost**: Machine learning model for classification
- **Librosa**: Audio feature extraction and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Model scaling and preprocessing

### Frontend
- **HTML/CSS**: User interface
- **Jinja2**: Template rendering

### Database
- **SQLite**: Relational database for data persistence

## 📁 Project Structure

```
lung-classification/
├── app.py                    # Main Flask application
├── healthcare.db             # SQLite database
├── xgboost_model1.pkl        # Trained XGBoost model
├── scaler.joblib            # Feature scaler
├── requirements.txt          # Python dependencies
├── runtime.txt              # Python version specification
├── procfile                 # Deployment configuration
├── static/
│   └── css/                 # Stylesheets
├── templates/               # HTML templates
│   ├── login.html
│   ├── signup.html
│   ├── index.html
│   ├── add_patient.html
│   ├── classify.html
│   ├── patient_details.html
│   └── patient_search.html
└── uploads/
    ├── photos/              # Patient photographs
    └── audio/               # Audio recordings
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/saikrishnamraju85131/lung-classification.git
   cd lung-classification
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   The database will be created automatically on first run.

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 💻 Usage

### Getting Started

1. **Create an Account**
   - Navigate to the signup page
   - Enter your doctor credentials
   - Login with your new account

2. **Add a Patient**
   - Click "Add Patient"
   - Fill in patient information:
     - First and last name
     - Age and gender
     - Contact phone number
     - Optional: Upload patient photo
   - Submit to create patient record

3. **Record and Classify**
   - Select a patient from your list
   - Upload a respiratory sound recording (WAV format)
   - System will:
     - Extract audio features
     - Process through XGBoost model
     - Display diagnosis with confidence score

4. **Review History**
   - Access patient details page
   - View all previous recordings and diagnoses
   - Track patient progress over time

### Audio Recording Guidelines

For best results:
- Use WAV format files
- Recording duration: 2.5+ seconds recommended
- Clear respiratory sounds without excessive background noise
- Position microphone consistently

## 🔒 Security Features

- **Password Hashing**: Werkzeug security for password protection
- **Session Management**: Secure user sessions with Flask
- **File Upload Validation**: Restricted file types to prevent malicious uploads
- **Secure Filenames**: Sanitized file naming to prevent path traversal attacks

## 🗄️ Database Schema

### Doctor
- `id`: Primary key
- `name`: Unique username
- `password`: Hashed password

### Patient
- `id`: Primary key
- `first_name`, `last_name`: Patient name
- `age`, `gender`: Demographics
- `phone`: Contact information
- `photo`: Profile picture path
- `registration_date`: Timestamp
- `doctor_id`: Foreign key to Doctor

### Recording
- `id`: Primary key
- `file_path`: Audio file location
- `recording_date`: Timestamp
- `patient_id`: Foreign key to Patient

### Diagnosis
- `id`: Primary key
- `result`: Classification result
- `confidence`: Prediction confidence (0-100%)
- `recording_id`: Foreign key to Recording

## 🎯 Model Information

### Algorithm: XGBoost
- **Type**: Gradient Boosting Classifier
- **Input**: 161 audio features extracted from respiratory sounds
- **Output**: 5-class classification (Bronchial, Asthma, COPD, Healthy, Pneumonia)
- **Preprocessing**: StandardScaler normalization

### Feature Engineering
The system extracts the following features from audio:
- **Zero-Crossing Rate**: Voice frequency characteristics
- **Chroma STFT**: Pitch class energy distribution
- **MFCC**: Timbral texture and sound quality
- **RMS Energy**: Audio intensity
- **Mel Spectrogram**: Frequency-domain representation

## 🌐 Deployment

### Heroku Deployment
The project includes configuration files for Heroku:
- `Procfile`: Gunicorn WSGI server configuration
- `runtime.txt`: Python version specification

To deploy:
```bash
heroku create your-app-name
git push heroku master
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page (redirects to login) |
| `/login` | GET, POST | Doctor login |
| `/signup` | GET, POST | Doctor registration |
| `/home` | GET | Dashboard |
| `/add_patient` | GET, POST | Add new patient |
| `/classify/<patient_id>` | GET, POST | Upload and classify audio |
| `/patient/<patient_id>` | GET | View patient details |
| `/search` | GET | Search patients |
| `/delete_diagnosis/<diagnosis_id>` | POST | Remove diagnosis |
| `/logout` | GET | End session |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is available for educational and research purposes.

## 👤 Author

**SAI KRISHNAMRAJU VEGESNA**
- GitHub: [@saikrishnamraju85131](https://github.com/saikrishnamraju85131)

## 🙏 Acknowledgments

- Audio processing powered by librosa
- Machine learning implementation using XGBoost
- Web framework built with Flask
- Medical domain knowledge from respiratory health research

## ⚠️ Disclaimer

**This application is for educational and research purposes only.** It should not be used as a replacement for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and treatment decisions.

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the repository maintainer

---

**Note**: This is an AI-assisted diagnostic tool designed to support, not replace, clinical judgment. All diagnoses should be verified by qualified medical professionals.
