import os
import io
import base64
import uuid
import logging
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
from flask_wtf import CSRFProtect

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_' + str(uuid.uuid4()))
csrf = CSRFProtect(app)

# Application name
APP_NAME = "MyHospital"

# Model paths
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
APPENDICITIS_MODEL_PATH = "app_best_model.pkl"

# Features for the model - updated to match the order in the training data
FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", 
    "ejection_fraction", "high_blood_pressure", "platelets", 
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

# Features for the appendicitis model - subset used in the form
APPENDICITIS_FEATURES = [
    "Age", "BMI", "Sex", "Migratory_Pain", "Lower_Right_Abd_Pain", 
    "Contralateral_Rebound_Tenderness", "Coughing_Pain", "Nausea", 
    "Loss_of_Appetite", "Body_Temperature", "WBC_Count", 
    "Neutrophil_Percentage", "Alvarado_Score", "Paedriatic_Appendicitis_Score"
]

# All columns needed by the appendicitis model
APPENDICITIS_ALL_COLUMNS = [
    "Age", "BMI", "Sex", "Height", "Weight", "Length_of_Stay", "Management", 
    "Severity", "Diagnosis_Presumptive", "Alvarado_Score", "Paedriatic_Appendicitis_Score", 
    "Appendix_on_US", "Appendix_Diameter", "Migratory_Pain", "Lower_Right_Abd_Pain", 
    "Contralateral_Rebound_Tenderness", "Coughing_Pain", "Nausea", "Loss_of_Appetite", 
    "Body_Temperature", "WBC_Count", "Neutrophil_Percentage", "Segmented_Neutrophils", 
    "Neutrophilia", "RBC_Count", "Hemoglobin", "RDW", "Thrombocyte_Count", 
    "Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine", "CRP", "Dysuria", 
    "Stool", "Peritonitis", "Psoas_Sign", "Ipsilateral_Rebound_Tenderness", 
    "US_Performed", "US_Number", "Free_Fluids", "Appendix_Wall_Layers", 
    "Target_Sign", "Appendicolith", "Perfusion", "Perforation", 
    "Surrounding_Tissue_Reaction", "Appendicular_Abscess", "Abscess_Location", 
    "Pathological_Lymph_Nodes", "Lymph_Nodes_Location", "Bowel_Wall_Thickening", 
    "Conglomerate_of_Bowel_Loops", "Ileus", "Coprostasis", "Meteorism", 
    "Enteritis", "Gynecological_Findings"
]

# Simulated doctor credentials - simplified
DOCTORS = {
    "admin@example.com": {
        "password": "admin123",
        "name": "Admin"
    }
}

# Valid ranges for numeric inputs with medical context
VALID_RANGES = {
    "age": (18, 100, "Years"),
    "sex": (0, 1, "Female/Male"),
    "anaemia": (0, 1, "Yes/No"),
    "creatinine_phosphokinase": (10, 8000, "U/L"),
    "diabetes": (0, 1, "Yes/No"),
    "ejection_fraction": (10, 80, "%"),
    "high_blood_pressure": (0, 1, "Yes/No"),
    "platelets": (50000, 500000, "per µL"),
    "serum_creatinine": (0.1, 10, "mg/dL"),
    "serum_sodium": (110, 150, "mmol/L"),
    "smoking": (0, 1, "Yes/No"),
    "time": (0, 365, "Days")
}

# Feature descriptions for medical context
FEATURE_INFO = {
    "age": "Patient's age in years",
    "sex": "Patient's gender (0 = Female, 1 = Male)",
    "anaemia": "Decrease of red blood cells or hemoglobin",
    "creatinine_phosphokinase": "Level of CPK enzyme in the blood (mcg/L)",
    "diabetes": "If the patient has diabetes",
    "ejection_fraction": "Percentage of blood leaving the heart at each contraction",
    "high_blood_pressure": "If the patient has hypertension",
    "platelets": "Platelets in the blood (kiloplatelets/mL)",
    "serum_creatinine": "Level of creatinine in the blood (mg/dL)",
    "serum_sodium": "Level of sodium in the blood (mEq/L)",
    "smoking": "If the patient smokes",
    "time": "Follow-up period (days)"
}

# Risk factors and their impact
RISK_FACTORS = {
    "age": "Older age increases risk of heart failure",
    "anaemia": "Reduces oxygen delivery, straining the heart",
    "creatinine_phosphokinase": "Elevated levels may indicate heart damage",
    "diabetes": "Damages blood vessels and increases heart strain",
    "ejection_fraction": "Lower values indicate weakened heart function",
    "high_blood_pressure": "Forces heart to work harder, causing strain",
    "platelets": "Abnormal levels may affect blood clotting",
    "serum_creatinine": "Elevated levels may indicate kidney dysfunction",
    "serum_sodium": "Imbalance can affect heart function",
    "smoking": "Damages blood vessels and reduces oxygen",
    "sex": "Men have a higher risk of heart failure than women"
}

# Valid ranges for appendicitis inputs
APPENDICITIS_RANGES = {
    "Age": (2, 18, "Years"),
    "BMI": (12, 40, "kg/m²"),
    "Sex": ("male", "female", "Gender"),
    "Migratory_Pain": (0, 1, "Yes/No"),
    "Lower_Right_Abd_Pain": (0, 1, "Yes/No"),
    "Contralateral_Rebound_Tenderness": (0, 1, "Yes/No"),
    "Coughing_Pain": (0, 1, "Yes/No"),
    "Nausea": (0, 1, "Yes/No"),
    "Loss_of_Appetite": (0, 1, "Yes/No"),
    "Body_Temperature": (35.0, 41.0, "°C"),
    "WBC_Count": (4000, 25000, "cells/μL"),
    "Neutrophil_Percentage": (20, 95, "%"),
    "Alvarado_Score": (0, 10, "Score"),
    "Paedriatic_Appendicitis_Score": (0, 10, "Score")
}

# Feature descriptions for appendicitis
APPENDICITIS_INFO = {
    "Age": "Patient's age in years",
    "BMI": "Body Mass Index (weight in kg / height in meters squared)",
    "Sex": "Patient's gender (male or female)",
    "Migratory_Pain": "Pain that moves from the umbilical region to the right lower quadrant",
    "Lower_Right_Abd_Pain": "Pain in the right lower abdomen",
    "Contralateral_Rebound_Tenderness": "Pain when pressing and quickly releasing the left side of the abdomen",
    "Coughing_Pain": "Pain when coughing",
    "Nausea": "Feeling of sickness with an inclination to vomit",
    "Loss_of_Appetite": "Reduced desire to eat",
    "Body_Temperature": "Body temperature in Celsius",
    "WBC_Count": "White blood cell count",
    "Neutrophil_Percentage": "Percentage of neutrophils in white blood cells",
    "Alvarado_Score": "Clinical scoring system used to diagnose appendicitis (0-10)",
    "Paedriatic_Appendicitis_Score": "Pediatric-specific scoring system for appendicitis (0-10)"
}

# Appendicitis risk factors
APPENDICITIS_RISK_FACTORS = {
    "Migratory_Pain": "Classic symptom of appendicitis as inflammation progresses",
    "Lower_Right_Abd_Pain": "Direct indication of appendix inflammation",
    "Contralateral_Rebound_Tenderness": "Indicates peritoneal irritation",
    "WBC_Count": "Elevated white blood cell count indicates infection",
    "Neutrophil_Percentage": "Elevated neutrophils suggest bacterial infection",
    "Body_Temperature": "Fever indicates inflammatory response",
    "Alvarado_Score": "Higher scores correlate with increased likelihood of appendicitis",
    "Paedriatic_Appendicitis_Score": "Pediatric-specific score with high predictive value"
}

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Model files not found: {e}. Please run the model training script first.")
    raise

# Load appendicitis model
try:
    appendicitis_model_dict = joblib.load(APPENDICITIS_MODEL_PATH)
    logging.info("Appendicitis model loaded successfully")
    # Extract the actual model from the dictionary
    if isinstance(appendicitis_model_dict, dict):
        appendicitis_model = appendicitis_model_dict.get('model', None)
        if appendicitis_model is None:
            # Try other common keys
            for key in ['best_model', 'classifier', 'estimator']:
                if key in appendicitis_model_dict:
                    appendicitis_model = appendicitis_model_dict[key]
                    break
        logging.info(f"Extracted appendicitis model: {type(appendicitis_model)}")
    else:
        appendicitis_model = appendicitis_model_dict
except FileNotFoundError as e:
    logging.error(f"Appendicitis model file not found: {e}. Please make sure the model file exists.")
    appendicitis_model = None
except Exception as e:
    logging.error(f"Error loading appendicitis model: {e}")
    appendicitis_model = None

# Generate interactive pie chart with Plotly
def generate_pie_chart(probability):
    fig = go.Figure(data=[go.Pie(
        labels=['Risk', 'No Risk'],
        values=[probability, 1-probability],
        hole=.4,
        marker_colors=['#FF5252', '#4CAF50'],
        textinfo='percent',
        textfont_size=14,
        textposition='inside'
    )])
    
    fig.update_layout(
        title={
            'text': 'Heart Failure Risk Assessment',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#333'}
        },
        annotations=[{
            'text': f"{probability*100:.1f}%<br>Risk",
            'x': 0.5, 'y': 0.5,
            'font_size': 20,
            'showarrow': False
        }],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        height=500,
        margin=dict(t=80, b=80, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# Generate radar chart for patient metrics
def generate_radar_chart(patient_data):
    # Select numeric features for the radar chart
    radar_features = ['age', 'ejection_fraction', 'serum_creatinine', 
                      'serum_sodium', 'creatinine_phosphokinase']
    
    # Get the values and normalize them to 0-1 range for radar chart
    values = []
    for feature in radar_features:
        min_val, max_val, _ = VALID_RANGES[feature]
        normalized_val = (patient_data[feature] - min_val) / (max_val - min_val)
        values.append(normalized_val)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=radar_features,
        fill='toself',
        name='Patient Metrics',
        line_color='#2196F3',
        fillcolor='rgba(33, 150, 243, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(t=60, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# Generate comparison chart
def generate_comparison_chart(patient_data):
    # Binary features to show as bar chart
    binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add binary features as bars
    fig.add_trace(
        go.Bar(
            x=binary_features,
            y=[patient_data[f] for f in binary_features],
            name="Risk Factors Present",
            marker_color='#FF5252',
            text=["Yes" if patient_data[f] == 1 else "No" for f in binary_features],
            textposition="auto"
        ),
        secondary_y=False,
    )
    
    # Add reference line
    fig.add_trace(
        go.Scatter(
            x=binary_features,
            y=[0.5, 0.5, 0.5, 0.5],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.2)", width=2, dash="dash"),
            name="Reference"
        ),
        secondary_y=False,
    )
    
    # Update layout
    fig.update_layout(
        title_text="Patient Risk Factors",
        height=350,
        margin=dict(t=60, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# Generate PDF report
def create_pdf_report(patient_data, prediction, probability):
    # Create a timestamp for the report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert probability to percentage
    risk_percentage = probability * 100
    
    # Generate pie chart for PDF
    plt.figure(figsize=(6, 6))
    labels = ['Risk', 'No Risk']
    sizes = [probability, 1-probability]
    colors = ['#FF5252', '#4CAF50']
    explode = (0.1, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Heart Failure Risk Assessment')
    
    # Save pie chart to base64 for embedding in PDF
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create HTML content for the PDF
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Heart Failure Risk Assessment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .logo {{ text-align: center; margin-bottom: 20px; }}
            h1 {{ color: #333366; }}
            .section {{ margin-bottom: 20px; }}
            .patient-info {{ border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .risk-high {{ color: #FF5252; font-weight: bold; }}
            .risk-low {{ color: #4CAF50; font-weight: bold; }}
            .footer {{ text-align: center; font-size: 12px; margin-top: 30px; color: #666; }}
            .chart-container {{ text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Heart Failure Risk Assessment Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Patient Information</h2>
            <div class="patient-info">
                <p><strong>Age:</strong> {patient_data['age']} years</p>
                <p><strong>Gender:</strong> {'Male' if patient_data['sex'] == 1 else 'Female'}</p>
                <p><strong>Diabetes:</strong> {'Yes' if patient_data['diabetes'] == 1 else 'No'}</p>
                <p><strong>Hypertension:</strong> {'Yes' if patient_data['high_blood_pressure'] == 1 else 'No'}</p>
                <p><strong>Smoking:</strong> {'Yes' if patient_data['smoking'] == 1 else 'No'}</p>
                <p><strong>Anemia:</strong> {'Yes' if patient_data['anaemia'] == 1 else 'No'}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Clinical Measurements</h2>
            <table>
                <tr>
                    <th>Measurement</th>
                    <th>Value</th>
                    <th>Normal Range</th>
                </tr>
                <tr>
                    <td>Ejection Fraction</td>
                    <td>{patient_data['ejection_fraction']}%</td>
                    <td>50-75%</td>
                </tr>
                <tr>
                    <td>Serum Creatinine</td>
                    <td>{patient_data['serum_creatinine']} mg/dL</td>
                    <td>0.6-1.2 mg/dL</td>
                </tr>
                <tr>
                    <td>Serum Sodium</td>
                    <td>{patient_data['serum_sodium']} mmol/L</td>
                    <td>135-145 mmol/L</td>
                </tr>
                <tr>
                    <td>Creatinine Phosphokinase</td>
                    <td>{patient_data['creatinine_phosphokinase']} U/L</td>
                    <td>10-120 U/L</td>
                </tr>
                <tr>
                    <td>Platelets</td>
                    <td>{patient_data['platelets']} per µL</td>
                    <td>150,000-450,000 per µL</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Risk Assessment</h2>
            <p>Based on the provided patient data, the heart failure risk is assessed as:
                <span class="{'risk-high' if prediction == 1 else 'risk-low'}">
                    {risk_percentage:.1f}% Risk {'(HIGH)' if prediction == 1 else '(LOW)'}
                </span>
            </p>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{img_str}" alt="Risk Assessment Chart" style="max-width: 100%;">
            </div>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
                {'<li>Immediate cardiology consultation is recommended.</li>' if prediction == 1 else ''}
                {'<li>Consider adjusting medications to improve ejection fraction.</li>' if patient_data['ejection_fraction'] < 40 else ''}
                {'<li>Monitor blood pressure regularly.</li>' if patient_data['high_blood_pressure'] == 1 else ''}
                {'<li>Diabetes management should be optimized.</li>' if patient_data['diabetes'] == 1 else ''}
                {'<li>Smoking cessation is strongly advised.</li>' if patient_data['smoking'] == 1 else ''}
                {'<li>Schedule follow-up appointment in 1-2 weeks.</li>' if prediction == 1 else '<li>Schedule routine follow-up in 3 months.</li>'}
                {'<li>Consider lifestyle modifications including diet and exercise.</li>'}
            </ul>
        </div>
        
        <div class="footer">
            <p>This report is generated automatically and should be reviewed by a healthcare professional.</p>
            <p> 2025 Heart Failure Prediction System</p>
        </div>
    </body>
    </html>
    '''
    
    # Create PDF from HTML
    pdf_output = io.BytesIO()
    pisa.CreatePDF(html_content, dest=pdf_output)
    pdf_output.seek(0)
    
    return pdf_output

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in DOCTORS and DOCTORS[email]['password'] == password:
            session['logged_in'] = True
            session['doctor_email'] = email
            session['doctor_name'] = DOCTORS[email]['name']
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid credentials. Please try again."
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    doctor = DOCTORS.get(session.get('username', ''))
    
    return render_template('dashboard.html', 
                           app_name=APP_NAME,
                           doctor=doctor,
                           prediction_features=[
                               {
                                   'name': 'Heart Failure Prediction',
                                   'description': 'Predict the risk of heart failure based on patient data',
                                   'icon': 'fa-heartbeat',
                                   'url': url_for('predict')
                               },
                               {
                                   'name': 'Pediatric Appendicitis Diagnosis',
                                   'description': 'Assess the risk of appendicitis in pediatric patients',
                                   'icon': 'fa-child',
                                   'url': url_for('appendicitis')
                               },
                               {
                                   'name': 'Predicting Success of Pediatric Bone Marrow Transplants',
                                   'description': 'Coming soon',
                                   'icon': 'fa-procedures',
                                   'url': '#'
                               },
                               {
                                   'name': 'Cervical Cancer Risk Assessment',
                                   'description': 'Coming soon',
                                   'icon': 'fa-venus',
                                   'url': '#'
                               },
                               {
                                   'name': 'Obesity Risk Estimation',
                                   'description': 'Coming soon',
                                   'icon': 'fa-weight',
                                   'url': '#'
                               }
                           ])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    prediction = None
    probability = None
    pie_chart = None
    radar_chart = None
    comparison_chart = None
    error = None
    form_values = {feature: "" for feature in FEATURES}
    
    if request.method == 'POST':
        try:
            # Process form data
            input_data = {}
            
            # Handle binary features (checkboxes)
            binary_features = ['sex', 'anaemia', 'diabetes', 'high_blood_pressure', 'smoking']
            for feature in binary_features:
                # If checkbox is checked, value will be in request.form, otherwise it's not present
                input_data[feature] = 1 if feature in request.form else 0
            
            # Handle numeric features
            numeric_features = [f for f in FEATURES if f not in binary_features]
            for feature in numeric_features:
                value = request.form.get(feature)
                if not value:
                    raise ValueError(f"Missing value for {feature}")
                
                value = float(value)
                min_val, max_val, _ = VALID_RANGES[feature]
                if value < min_val or value > max_val:
                    raise ValueError(f"{feature.replace('_', ' ').title()} must be between {min_val} and {max_val}")
                input_data[feature] = value
            
            # Store form values for redisplay if needed
            form_values = input_data
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_data])
            
            # Ensure the order of features matches what the model expects
            input_df = input_df[FEATURES]
            
            # Scale input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = int(model.predict(input_scaled)[0])
            probability = float(model.predict_proba(input_scaled)[0, 1])
            
            # Generate charts
            pie_chart = generate_pie_chart(probability)
            radar_chart = generate_radar_chart(input_data)
            comparison_chart = generate_comparison_chart(input_data)
            
            # Store prediction in session for PDF generation
            session['last_prediction'] = {
                'input_data': input_data,
                'prediction': prediction,
                'probability': probability
            }
            
            logging.info(f"Prediction made: {prediction}, Probability: {probability}")
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            error = str(e)
    
    return render_template('predict.html', 
                           features=FEATURES,
                           feature_info=FEATURE_INFO,
                           valid_ranges=VALID_RANGES,
                           form_values=form_values,
                           prediction=prediction,
                           probability=probability,
                           pie_chart=pie_chart,
                           radar_chart=radar_chart,
                           comparison_chart=comparison_chart,
                           error=error,
                           risk_factors=RISK_FACTORS)

@app.route('/generate_report')
def generate_report():
    if not session.get('logged_in') or 'last_prediction' not in session:
        return redirect(url_for('predict'))
    
    # Get prediction data from session
    pred_data = session['last_prediction']
    
    # Create PDF report
    pdf_output = create_pdf_report(
        pred_data['input_data'],
        pred_data['prediction'],
        pred_data['probability']
    )
    
    # Generate a filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"heart_failure_risk_report_{timestamp}.pdf"
    
    # Send the PDF as a downloadable file
    return send_file(
        pdf_output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

@app.route('/appendicitis', methods=['GET', 'POST'])
def appendicitis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if appendicitis_model is None:
        return render_template('error.html', error="Appendicitis model not loaded. Please contact the administrator.")
    
    prediction = None
    probability = None
    pie_chart = None
    radar_chart = None
    error = None
    form_values = {feature: "" for feature in APPENDICITIS_FEATURES}
    
    if request.method == 'POST':
        try:
            # Process form data
            input_data = {}
            
            # Handle binary features (checkboxes)
            binary_features = ['Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness', 
                              'Coughing_Pain', 'Nausea', 'Loss_of_Appetite']
            for feature in binary_features:
                # If checkbox is checked, value will be in request.form, otherwise it's not present
                input_data[feature] = 1 if feature in request.form else 0
            
            # Handle sex (special case)
            input_data['Sex'] = request.form.get('Sex')
            
            # Handle numeric features
            numeric_features = [f for f in APPENDICITIS_FEATURES if f not in binary_features and f != 'Sex']
            for feature in numeric_features:
                value = request.form.get(feature)
                if not value:
                    raise ValueError(f"Missing value for {feature}")
                
                value = float(value)
                if feature in APPENDICITIS_RANGES:
                    min_val, max_val, _ = APPENDICITIS_RANGES[feature]
                    if value < min_val or value > max_val:
                        raise ValueError(f"{feature.replace('_', ' ').title()} must be between {min_val} and {max_val}")
                input_data[feature] = value
            
            # Store form values for redisplay if needed
            form_values = input_data
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_data])
            
            # Add missing columns with default values
            for col in APPENDICITIS_ALL_COLUMNS:
                if col not in input_df.columns:
                    if col in ['Height', 'Weight', 'Length_of_Stay', 'WBC_Count', 
                              'Neutrophil_Percentage', 'Segmented_Neutrophils', 
                              'RBC_Count', 'Hemoglobin', 'RDW', 'Thrombocyte_Count', 
                              'CRP', 'Appendix_Diameter']:
                        # Numeric columns get 0 as default
                        input_df[col] = 0
                    elif col in ['Sex']:
                        # Skip Sex as it's already handled
                        continue
                    elif col in ['Diagnosis_Presumptive']:
                        # This is what we're predicting, so use a neutral value
                        input_df[col] = 'unknown'
                    else:
                        # Binary or categorical columns get 'no' or 0 as default
                        input_df[col] = 0
            
            # Make sure columns are in the right order
            input_df = input_df[APPENDICITIS_ALL_COLUMNS]
            
            # Make prediction
            try:
                if hasattr(appendicitis_model, 'predict'):
                    # If we have a direct model object
                    prediction = int(appendicitis_model.predict(input_df)[0])
                    probability = float(appendicitis_model.predict_proba(input_df)[0, 1])
                elif isinstance(appendicitis_model_dict, dict) and 'predict_function' in appendicitis_model_dict:
                    # If we have a dictionary with a predict_function
                    predictions, probabilities = appendicitis_model_dict['predict_function'](input_df)
                    prediction = int(predictions[0])
                    probability = float(probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0])
                else:
                    raise ValueError("Could not find a valid prediction method in the model")
                
                logging.info(f"Prediction made successfully: {prediction}, Probability: {probability}")
            except Exception as e:
                logging.error(f"Error during prediction: {str(e)}")
                raise ValueError(f"Error making prediction: {str(e)}")
            
            # Generate charts
            pie_chart = generate_appendicitis_pie_chart(probability)
            radar_chart = generate_appendicitis_radar_chart(input_data)
            
            # Store prediction in session for PDF generation
            session['last_appendicitis_prediction'] = {
                'input_data': input_data,
                'prediction': prediction,
                'probability': probability
            }
            
            logging.info(f"Appendicitis prediction made: {prediction}, Probability: {probability}")
            
        except Exception as e:
            logging.error(f"Error during appendicitis prediction: {str(e)}")
            error = str(e)
    
    return render_template('appendicitis.html', 
                           features=APPENDICITIS_FEATURES,
                           feature_info=APPENDICITIS_INFO,
                           valid_ranges=APPENDICITIS_RANGES,
                           form_values=form_values,
                           prediction=prediction,
                           probability=probability,
                           pie_chart=pie_chart,
                           radar_chart=radar_chart,
                           error=error,
                           risk_factors=APPENDICITIS_RISK_FACTORS)

# Generate appendicitis pie chart
def generate_appendicitis_pie_chart(probability):
    fig = go.Figure(data=[go.Pie(
        labels=['Appendicitis', 'No Appendicitis'],
        values=[probability, 1-probability],
        hole=.4,
        marker_colors=['#FF5252', '#4CAF50'],
        textinfo='percent',
        textfont_size=14,
        textposition='inside'
    )])
    
    fig.update_layout(
        title_text="Appendicitis Risk Assessment",
        annotations=[dict(text=f"{probability*100:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# Generate appendicitis radar chart
def generate_appendicitis_radar_chart(patient_data):
    # Select key features for radar chart
    radar_features = ['Age', 'BMI', 'WBC_Count', 'Neutrophil_Percentage', 
                      'Alvarado_Score', 'Paedriatic_Appendicitis_Score']
    
    # Normalize values for radar chart (0-1 scale)
    normalized_values = []
    for feature in radar_features:
        if feature in patient_data:
            value = patient_data[feature]
            if feature in APPENDICITIS_RANGES:
                min_val, max_val, _ = APPENDICITIS_RANGES[feature]
                normalized = (value - min_val) / (max_val - min_val)
                normalized_values.append(normalized)
            else:
                normalized_values.append(0.5)  # Default if no range
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=radar_features,
        fill='toself',
        name='Patient Data',
        line_color='rgb(31, 119, 180)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Patient Metrics",
        showlegend=False,
        height=350,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

@app.route('/generate_appendicitis_report')
def generate_appendicitis_report():
    if not session.get('logged_in') or not session.get('last_appendicitis_prediction'):
        return redirect(url_for('login'))
    
    # Get prediction data from session
    prediction_data = session.get('last_appendicitis_prediction')
    patient_data = prediction_data.get('input_data', {})
    prediction = prediction_data.get('prediction', 0)
    probability = prediction_data.get('probability', 0)
    
    # Create PDF report
    pdf_output = create_appendicitis_pdf_report(patient_data, prediction, probability)
    
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"appendicitis_risk_report_{timestamp}.pdf"
    
    # Send the PDF as a downloadable file
    return send_file(
        pdf_output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

def create_appendicitis_pdf_report(patient_data, prediction, probability):
    # Create a timestamp for the report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate pie chart for PDF
    plt.figure(figsize=(6, 6))
    labels = ['Appendicitis', 'No Appendicitis']
    sizes = [probability * 100, (1 - probability) * 100]
    colors = ['#FF5252', '#4CAF50']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Appendicitis Risk Assessment')
    
    # Save pie chart to base64 for embedding in PDF
    pie_chart_buffer = io.BytesIO()
    plt.savefig(pie_chart_buffer, format='png')
    plt.close()
    pie_chart_base64 = base64.b64encode(pie_chart_buffer.getvalue()).decode('utf-8')
    
    # Create HTML content for the PDF
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .risk-high {{ color: #FF5252; }}
            .risk-low {{ color: #4CAF50; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #777; text-align: center; }}
            .chart-container {{ text-align: center; margin: 20px 0; }}
            .risk-factors {{ margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Pediatric Appendicitis Risk Assessment Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <h2>Risk Assessment Result</h2>
        <p>Based on the provided patient data, the risk of appendicitis is:
            <strong class="{'risk-high' if prediction == 1 else 'risk-low'}">
                {f"High ({probability*100:.1f}%)" if prediction == 1 else f"Low ({(1-probability)*100:.1f}%)"}
            </strong>
        </p>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{pie_chart_base64}" alt="Risk Assessment Chart" width="400">
        </div>
        
        <h2>Patient Data</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
    """
    
    # Add patient data to the table
    for feature, value in patient_data.items():
        if feature in APPENDICITIS_INFO:
            description = APPENDICITIS_INFO[feature]
            
            # Format binary values
            if feature in ['Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness', 
                          'Coughing_Pain', 'Nausea', 'Loss_of_Appetite']:
                value = "Yes" if value == 1 else "No"
                
            html_content += f"""
            <tr>
                <td>{feature.replace('_', ' ').title()}</td>
                <td>{value}</td>
                <td>{description}</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <div class="risk-factors">
            <h2>Key Risk Factors for Appendicitis</h2>
            <ul>
    """
    
    # Add risk factors
    for factor, description in APPENDICITIS_RISK_FACTORS.items():
        html_content += f"<li><strong>{factor.replace('_', ' ').title()}:</strong> {description}</li>"
    
    html_content += """
            </ul>
        </div>
        
        <div class="footer">
            <p>This report is generated by MyHospital Medical Prediction System. This is not a definitive diagnosis and should be used in conjunction with clinical judgment.</p>
        </div>
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    pdf_output = io.BytesIO()
    pisa.CreatePDF(html_content, dest=pdf_output)
    pdf_output.seek(0)
    
    return pdf_output

# Ensure static directory exists
def create_static_dir():
    os.makedirs('static', exist_ok=True)

# Call the function to create static directory
create_static_dir()

if __name__ == '__main__':
    app.run(debug=True)
