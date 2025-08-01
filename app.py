from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import pickle
import numpy as np
from io import BytesIO
from xhtml2pdf import pisa
from symptom_mapper import SYMPTOM_SYNONYMS

app = Flask(__name__)

# Load model and encoders
mlp_model = pickle.load(open("model_mlp.pkl", "rb"))
trained_symptoms = pickle.load(open("symptom_encoder.pkl", "rb"))
disease_encoder = pickle.load(open("disease_encoder.pkl", "rb"))

# Load dataset
df = pd.read_csv("dataset/dataset.csv")
symptom_cols = [col for col in df.columns if col != "Disease"]
df[symptom_cols] = df[symptom_cols].fillna("none").astype(str)
all_symptoms = sorted(set(df[symptom_cols].values.ravel()) - {"none"})

# Supporting files
desc_df = pd.read_csv("data/symptom_Description.csv")
precaution_df = pd.read_csv("data/symptom_precaution.csv")
med_test_df = pd.read_csv("data/med_test_clean.csv")

last_result_data = {}

@app.route('/')
def index():
    return render_template("index.html", symptoms=sorted(all_symptoms))

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get("name")
    age = request.form.get("age")
    gender = request.form.get("gender")
    method = request.form.get("method", "mlp")

    selected_symptoms = [request.form.get(f"symptom{i}") for i in range(1, 6)]
    selected_symptoms = [s for s in selected_symptoms if s]

    typed = request.form.get("typed_symptoms", "")
    typed_symptoms = [s.strip().lower() for s in typed.split(",") if s.strip()]

    for term in typed_symptoms:
        mapped = SYMPTOM_SYNONYMS.get(term, term.replace(" ", "_"))
        if mapped not in selected_symptoms:
            selected_symptoms.append(mapped)

    if not selected_symptoms:
        return "Please select or enter at least one symptom.", 400

    return predict_logic(name, age, gender, selected_symptoms, method)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    name = data.get("name", "Anonymous")
    age = data.get("age", "0")
    gender = data.get("gender", "N/A")
    method = data.get("method", "mlp")

    typed_symptoms = data.get("typed_symptoms", "")
    if isinstance(typed_symptoms, str):
        symptoms = [s.strip().lower() for s in typed_symptoms.split(",") if s.strip()]
    elif isinstance(typed_symptoms, list):
        symptoms = [s.strip().lower() for s in typed_symptoms]
    else:
        return jsonify({"error": "Invalid symptom format"}), 400

    selected_symptoms = []
    for term in symptoms:
        mapped = SYMPTOM_SYNONYMS.get(term, term.replace(" ", "_"))
        if mapped not in selected_symptoms:
            selected_symptoms.append(mapped)

    if not selected_symptoms:
        return jsonify({"error": "No valid symptoms provided"}), 400

    result = predict_logic(name, age, gender, selected_symptoms, method, return_json=True)
    return jsonify(result)

def predict_logic(name, age, gender, selected_symptoms, method="mlp", return_json=False):
    input_vector = np.array([1 if s in selected_symptoms else 0 for s in trained_symptoms]).reshape(1, -1)
    top_diseases = []

    if method == "mlp":
        probs = mlp_model.predict_proba(input_vector)[0]
        top_indices = probs.argsort()[-3:][::-1]
        for idx in top_indices:
            disease = disease_encoder.inverse_transform([idx])[0]
            confidence = round(float(probs[idx]) * 100, 2)
            top_diseases.append([disease, confidence])
    else:
        vectors = df[symptom_cols].apply(lambda row: [1 if s in row.values else 0 for s in trained_symptoms], axis=1, result_type="expand").values
        diseases = df["Disease"].values
        sims = [np.sum(np.logical_and(input_vector[0], vec)) / np.sum(np.logical_or(input_vector[0], vec)) if np.sum(np.logical_or(input_vector[0], vec)) > 0 else 0 for vec in vectors]
        sim_df = pd.DataFrame({"Disease": diseases, "Similarity": sims})
        sim_df = sim_df.groupby("Disease")["Similarity"].max().reset_index()
        sim_df = sim_df.sort_values(by="Similarity", ascending=False).head(3)
        top_diseases = [[row["Disease"], round(float(row["Similarity"]) * 100, 2)] for _, row in sim_df.iterrows()]

    predicted_disease = top_diseases[0][0]

    # Description
    description = "N/A"
    desc_row = desc_df[desc_df["Disease"] == predicted_disease]
    if not desc_row.empty:
        description = desc_row["Description"].values[0]

    # Precautions
    precautions = []
    if predicted_disease in precaution_df["Disease"].values:
        row = precaution_df[precaution_df["Disease"] == predicted_disease].iloc[0]
        precautions = [row.get(f"Precaution_{i}", "N/A") for i in range(1, 5)]

    # Medicines and Tests
    medicines, tests = [], []
    row = med_test_df[med_test_df["Disease"] == predicted_disease]
    if not row.empty:
        row = row.iloc[0]
        if pd.notna(row["Medicines"]):
            medicines = row["Medicines"].split(";")
        if pd.notna(row["Recommended_Tests"]):
            tests = row["Recommended_Tests"].split(";")

    global last_result_data
    last_result_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "selected_symptoms": [s.replace("_", " ").title() for s in selected_symptoms],
        "disease": predicted_disease,
        "description": description,
        "top_diseases": top_diseases,
        "precautions": precautions,
        "medicines": medicines,
        "tests": tests
    }

    if return_json:
        return last_result_data
    else:
        return render_template("result.html", **last_result_data)

@app.route('/download_pdf')
def download_pdf():
    if not last_result_data:
        return "No data to generate PDF.", 400

    html = render_template("result.html", **last_result_data)
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf)
    pdf.seek(0)

    if pisa_status.err:
        return "Error generating PDF", 500

    return send_file(pdf, download_name="diagnosis_result.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
