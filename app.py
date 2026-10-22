from flask import Flask, render_template_string, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained Support Vector Classifier model
model = joblib.load("svcm.pkl")

# Modern, responsive UI layout with embedded style code
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction</title>
    <style>
        :root {
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --border-color: #d1d5db;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            width: 100%;
            max-width: 750px;
        }
        h2 {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 25px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 14px;
        }
        input, select {
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result {
            margin-top: 25px;
            padding: 15px;
            background-color: #e0e7ff;
            color: #312e81;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 18px;
        }
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            button {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Health Risk Assessment Dashboard</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" required>
                </div>
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select id="gender" name="gender" required>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="city">City Code</label>
                    <input type="number" id="city" name="city" required>
                </div>
                <div class="form-group">
                    <label for="bmi">BMI</label>
                    <input type="number" step="0.1" id="bmi" name="bmi" required>
                </div>
                <div class="form-group">
                    <label for="family_history_diabetes">Family History of Diabetes</label>
                    <select id="family_history_diabetes" name="family_history_diabetes" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="physical_activity_level">Physical Activity Level</label>
                    <select id="physical_activity_level" name="physical_activity_level" required>
                        <option value="0">Low</option>
                        <option value="1">Moderate</option>
                        <option value="2">High</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="diet_type">Diet Type</label>
                    <select id="diet_type" name="diet_type" required>
                        <option value="0">Standard</option>
                        <option value="1">Vegetarian</option>
                        <option value="2">Vegan</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="smoking_status">Smoking Status</label>
                    <select id="smoking_status" name="smoking_status" required>
                        <option value="0">Non-Smoker</option>
                        <option value="1">Former Smoker</option>
                        <option value="2">Current Smoker</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="alcohol_consumption">Alcohol Consumption</label>
                    <select id="alcohol_consumption" name="alcohol_consumption" required>
                        <option value="0">None</option>
                        <option value="1">Low</option>
                        <option value="2">Moderate/High</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="hours_sleep_per_night">Hours Sleep Per Night</label>
                    <input type="number" step="0.1" id="hours_sleep_per_night" name="hours_sleep_per_night" required>
                </div>
                <div class="form-group">
                    <label for="stress_level">Stress Level (1-10)</label>
                    <input type="number" id="stress_level" name="stress_level" required>
                </div>
                <div class="form-group">
                    <label for="fasting_blood_sugar">Fasting Blood Sugar</label>
                    <input type="number" step="0.1" id="fasting_blood_sugar" name="fasting_blood_sugar" required>
                </div>
                <div class="form-group">
                    <label for="hba1c_level">HbA1c Level</label>
                    <input type="number" step="0.1" id="hba1c_level" name="hba1c_level" required>
                </div>
                <div class="form-group">
                    <label for="blood_pressure_systolic">BP Systolic</label>
                    <input type="number" id="blood_pressure_systolic" name="blood_pressure_systolic" required>
                </div>
                <div class="form-group">
                    <label for="blood_pressure_diastolic">BP Diastolic</label>
                    <input type="number" id="blood_pressure_diastolic" name="blood_pressure_diastolic" required>
                </div>
                <div class="form-group">
                    <label for="waist_circumference_cm">Waist Circumference (cm)</label>
                    <input type="number" step="0.1" id="waist_circumference_cm" name="waist_circumference_cm" required>
                </div>
                <div class="form-group">
                    <label for="income_bracket">Income Bracket</label>
                    <input type="number" id="income_bracket" name="income_bracket" required>
                </div>
                <button type="submit">Predict Result</button>
            </div>
        </form>
        {% if prediction is not none %}
        <div class="result">
            Prediction Result: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Order matches features found in model structure
            features = [
                float(request.form["age"]),
                float(request.form["gender"]),
                float(request.form["city"]),
                float(request.form["bmi"]),
                float(request.form["family_history_diabetes"]),
                float(request.form["physical_activity_level"]),
                float(request.form["diet_type"]),
                float(request.form["smoking_status"]),
                float(request.form["alcohol_consumption"]),
                float(request.form["hours_sleep_per_night"]),
                float(request.form["stress_level"]),
                float(request.form["fasting_blood_sugar"]),
                float(request.form["hba1c_level"]),
                float(request.form["blood_pressure_systolic"]),
                float(request.form["blood_pressure_diastolic"]),
                float(request.form["waist_circumference_cm"]),
                float(request.form["income_bracket"])
            ]
            
            pred_array = np.array([features])
            prediction_value = model.predict(pred_array)[0]
            prediction = f"Positive (Class {prediction_value})" if prediction_value == 1 else f"Negative (Class {prediction_value})"
        except Exception as e:
            prediction = f"Error processing input: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
