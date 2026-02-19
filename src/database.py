import sqlite3
import pandas as pd
from datetime import datetime
import json
import os

class CardioAIDB:
    def __init__(self, db_path='data/database.db'):
        self.db_path = db_path
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                age INTEGER,
                sex INTEGER,
                cholesterol_total REAL,
                cholesterol_hdl REAL,
                cholesterol_ldl REAL,
                triglycerides REAL,
                blood_pressure_systolic INTEGER,
                blood_pressure_diastolic INTEGER,
                glucose REAL,
                bmi REAL,
                smoking INTEGER,
                physical_activity INTEGER,
                risk_score INTEGER,
                risk_category TEXT,
                recommendations TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Doctor-Patient relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctor_patients (
                doctor_id INTEGER,
                patient_id INTEGER,
                PRIMARY KEY (doctor_id, patient_id),
                FOREIGN KEY (doctor_id) REFERENCES users (id),
                FOREIGN KEY (patient_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, email, name, role='patient'):
        """Add new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (email, name, role) VALUES (?, ?, ?)',
                (email, name, role)
            )
            conn.commit()
            user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            user_id = cursor.fetchone()[0]
        finally:
            conn.close()
        return user_id
    
    def add_assessment(self, user_id, data):
        """Add new assessment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO assessments (
                user_id, age, sex, cholesterol_total, cholesterol_hdl,
                cholesterol_ldl, triglycerides, blood_pressure_systolic,
                blood_pressure_diastolic, glucose, bmi, smoking,
                physical_activity, risk_score, risk_category, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, data['age'], data['sex'], data['cholesterol_total'],
            data['cholesterol_hdl'], data['cholesterol_ldl'], data['triglycerides'],
            data['blood_pressure_systolic'], data['blood_pressure_diastolic'],
            data['glucose'], data['bmi'], data['smoking'], data['physical_activity'],
            data['risk_score'], data['risk_category'], json.dumps(data['recommendations'])
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id):
        """Get assessment history for a user"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT * FROM assessments WHERE user_id = ? ORDER BY assessment_date DESC',
            conn, params=(user_id,)
        )
        conn.close()
        return df
    
    def get_all_patients(self):
        """Get all patients with latest assessment"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT 
                u.id, u.name, u.email,
                a.risk_score, a.risk_category, a.assessment_date
            FROM users u
            LEFT JOIN (
                SELECT user_id, risk_score, risk_category, assessment_date,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY assessment_date DESC) as rn
                FROM assessments
            ) a ON u.id = a.user_id AND a.rn = 1
            WHERE u.role = 'patient'
            ORDER BY a.risk_score DESC NULLS LAST
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_population_stats(self):
        """Get population health statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Latest assessment for each patient
        query = '''
            SELECT 
                risk_category,
                AVG(age) as avg_age,
                AVG(bmi) as avg_bmi,
                AVG(cholesterol_total) as avg_cholesterol,
                COUNT(*) as count
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY assessment_date DESC) as rn
                FROM assessments
            )
            WHERE rn = 1
            GROUP BY risk_category
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
