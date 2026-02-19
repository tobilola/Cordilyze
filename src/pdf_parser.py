import PyPDF2
import pdfplumber
import re

class LabReportParser:
    def __init__(self):
        self.patterns = {
            'cholesterol_total': r'(?:total\s+)?cholesterol[:\s]+(\d+(?:\.\d+)?)',
            'cholesterol_hdl': r'hdl[:\s]+(\d+(?:\.\d+)?)',
            'cholesterol_ldl': r'ldl[:\s]+(\d+(?:\.\d+)?)',
            'triglycerides': r'triglycerides[:\s]+(\d+(?:\.\d+)?)',
            'glucose': r'glucose[:\s]+(\d+(?:\.\d+)?)',
            'blood_pressure': r'(?:bp|blood\s+pressure)[:\s]+(\d+)/(\d+)',
        }
    
    def extract_text(self, pdf_path):
        """Extract text from PDF"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except:
            # Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        
        return text.lower()
    
    def parse_lab_values(self, pdf_path):
        """Parse lab values from PDF"""
        text = self.extract_text(pdf_path)
        
        results = {}
        
        # Extract cholesterol values
        for key, pattern in self.patterns.items():
            if key == 'blood_pressure':
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    results['blood_pressure_systolic'] = int(match.group(1))
                    results['blood_pressure_diastolic'] = int(match.group(2))
            else:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    results[key] = float(match.group(1))
        
        return results
    
    def validate_values(self, values):
        """Validate extracted values are in reasonable ranges"""
        ranges = {
            'cholesterol_total': (100, 400),
            'cholesterol_hdl': (20, 100),
            'cholesterol_ldl': (50, 250),
            'triglycerides': (50, 500),
            'glucose': (70, 250),
            'blood_pressure_systolic': (90, 200),
            'blood_pressure_diastolic': (60, 130),
        }
        
        validated = {}
        warnings = []
        
        for key, value in values.items():
            if key in ranges:
                min_val, max_val = ranges[key]
                if min_val <= value <= max_val:
                    validated[key] = value
                else:
                    warnings.append(f"{key}: {value} is outside expected range ({min_val}-{max_val})")
        
        return validated, warnings
