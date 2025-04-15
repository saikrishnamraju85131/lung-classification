from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
from joblib import load
import pickle
import librosa
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PHOTO_FOLDER = os.path.join(UPLOAD_FOLDER, 'photos')
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PHOTO_FOLDER'] = PHOTO_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'healthcare.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav'}

# Ensure upload folders exist
os.makedirs(PHOTO_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# Models
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    patients = db.relationship('Patient', backref='doctor', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    photo = db.Column(db.String(200))
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    recordings = db.relationship('Recording', backref='patient', lazy=True)

class Recording(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.String(200), nullable=False)
    recording_date = db.Column(db.DateTime, default=datetime.utcnow)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    diagnosis = db.relationship('Diagnosis', backref='recording', uselist=False)

class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(data):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_features(file_path, scaler):
    try:
        data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)
        features = extract_features(data)
        return scaler.transform(features.reshape(1, -1)).flatten()
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Routes
@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        doctor = Doctor.query.filter_by(name=name).first()
        
        if doctor and check_password_hash(doctor.password, password):
            session['user'] = doctor.name
            return redirect(url_for('home'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        password = generate_password_hash(request.form['password'])
        
        if Doctor.query.filter_by(name=name).first():
            flash('Username already exists', 'error')
        else:
            doctor = Doctor(name=name, password=password)
            db.session.add(doctor)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        first_name = request.form['firstName']
        last_name = request.form['lastName']
        age = request.form['age']
        gender = request.form['gender']
        phone = request.form['phone']
        
        photo = request.files.get('photo')
        photo_path = None
        if photo and photo.filename != '':
            if allowed_file(photo.filename):
                try:
                    filename = secure_filename(photo.filename)
                    unique_filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{filename}"
                    # Ensure upload directory exists
                    os.makedirs(app.config['PHOTO_FOLDER'], exist_ok=True)
                    photo_path = os.path.join('photos', unique_filename)
                    photo_path = photo_path.replace('\\', '/')
                    full_path = os.path.join(app.config['PHOTO_FOLDER'], unique_filename)
                    photo.save(full_path)
                    if not os.path.exists(full_path):
                        raise Exception("File failed to save")
                except Exception as e:
                    flash(f"Error saving photo: {str(e)}", "error")
                    photo_path = None
            else:
                flash("Invalid file type for photo", "error")
            print(photo_path)            

        
        doctor = Doctor.query.filter_by(name=session['user']).first()
        patient = Patient(
            first_name=first_name,
            last_name=last_name,
            age=age,
            gender=gender,
            phone=phone,
            photo=photo_path,
            doctor_id=doctor.id
        )
        
        db.session.add(patient)
        db.session.commit()
        flash('Patient added successfully!', 'success')
        return redirect(url_for('classify', patient_id=patient.id))
    
    return render_template('add_patient.html')

@app.route('/classify/<int:patient_id>', methods=['GET', 'POST'])
def classify(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    patient = Patient.query.get_or_404(patient_id)
    
    if request.method == 'POST':
        audio = request.files.get('audio')
        if not audio or not allowed_file(audio.filename):
            flash('Invalid audio file', 'error')
            return redirect(url_for('classify', patient_id=patient_id))
        
        filename = secure_filename(audio.filename)
        audio_path = os.path.join('audio', filename)
        audio.save(os.path.join(app.config['AUDIO_FOLDER'], filename))
        
        # Save recording
        recording = Recording(
            file_path=audio_path,
            patient_id=patient_id
        )
        db.session.add(recording)
        
        # Process and predict
        try:
            scaler = load('scaler.joblib')
            model = pickle.load(open('xgboost_model1.pkl', 'rb'))
            features = get_features(os.path.join(app.config['AUDIO_FOLDER'], filename), scaler)
            
            if features is not None:
                prediction = model.predict([features])[0]
                proba = model.predict_proba([features])[0].max()
                prediction_map = {0: 'Bronchial', 1: 'Asthma', 2: 'COPD', 3: 'Healthy', 4: 'Pneumonia'}
                result = prediction_map.get(prediction, 'Unknown')
                
                diagnosis = Diagnosis(
                    result=result,
                    confidence=round(proba * 100, 2),
                    recording=recording
                )
                db.session.add(diagnosis)
                db.session.commit()
                
                flash(f'Diagnosis complete: {result} ({diagnosis.confidence}% confidence)', 'success')
                return redirect(url_for('patient_details', patient_id=patient_id))
        
        except Exception as e:
            db.session.rollback()
            flash(f'Error during classification: {str(e)}', 'error')
        
    return render_template('classify.html', patient=patient)

@app.route('/patient/<int:patient_id>')
def patient_details(patient_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    patient = Patient.query.get_or_404(patient_id)
    print("DEBUG: Photo path stored in DB =", patient.photo) 
    return render_template('patient_details.html', patient=patient)

@app.route('/search')
def search_patients():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    query = request.args.get('query', '')
    doctor = Doctor.query.filter_by(name=session['user']).first()
    patients = Patient.query.filter(
        (Patient.first_name.ilike(f'%{query}%')) | 
        (Patient.last_name.ilike(f'%{query}%')) |
        (Patient.phone.ilike(f'%{query}%')),
        Patient.doctor_id == doctor.id
    ).all()
    return render_template('patient_search.html', patients=patients)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.context_processor
def utility_processor():
    def file_exists(path):
        return os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], path))
    return dict(file_exists=file_exists)

@app.route('/delete_diagnosis/<int:diagnosis_id>', methods=['POST'])
def delete_diagnosis(diagnosis_id):
    diagnosis = Diagnosis.query.get_or_404(diagnosis_id)
    
    # Optional: Store patient ID to redirect after deletion
    recording = diagnosis.recording
    patient_id = recording.patient_id

    db.session.delete(diagnosis)
    db.session.commit()

    flash("Diagnosis deleted successfully!", "success")
    return redirect(url_for('patient_details', patient_id=patient_id))




@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)