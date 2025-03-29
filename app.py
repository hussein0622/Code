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

# Features for the model - updated to match the order in the training data
FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", 
    "ejection_fraction", "high_blood_pressure", "platelets", 
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
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

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Model files not found: {e}. Please run the model training script first.")
    raise

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
    
    return render_template('dashboard.html', 
                           doctor_name=session.get('doctor_name', 'Doctor'))

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

# Ensure static directory exists
def create_static_dir():
    os.makedirs('static', exist_ok=True)

# Call the function to create static directory
create_static_dir()

if __name__ == '__main__':
    app.run(debug=True)
