pip install Flask
from flask import Flask, render_template, request
import csv
import os

app = Flask(__name__)

# Ensure the 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

@app.route('/')
def home():
    return render_template('index.html')  # The page with the form

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = {
        'age': request.form['age'],
        'anaemia': request.form['anaemia'],
        'creatinine_phosphokinase': request.form['creatinine_phosphokinase'],
        'diabetes': request.form['diabetes'],
        'ejection_fraction': request.form['ejection_fraction'],
        'high_blood_pressure': request.form['high_blood_pressure'],
        'platelets': request.form['platelets'],
        'serum_creatinine': request.form['serum_creatinine'],
        'serum_sodium': request.form['serum_sodium'],
        'sex': request.form['sex'],
        'smoking': request.form['smoking'],
        'time': request.form['time'],
        'name': request.form['name']
    }

    # Define the CSV file path inside the 'data' folder
    file_path = 'data/prediction_data.csv'
    
    # Check if the file exists, and if not, create it and write the header
    try:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if file.tell() == 0:  # If file is empty, write the header
                writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        return f"An error occurred while saving the data: {e}"

    # Prediction logic (just an example of how you might handle predictions)
    # In a real scenario, you can load a trained model and make predictions here
    prediction = "Prediction Result: Positive"  # Replace with actual model prediction logic

    # Render the result page with the prediction
    return
