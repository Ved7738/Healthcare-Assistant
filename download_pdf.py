from fpdf import FPDF
import os

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_text_color(0)
    pdf.cell(200, 10, txt=f"Prediction Result for {data['name']}", ln=True, align="C")

    pdf.set_text_color(255, 0, 0)
    pdf.cell(200, 10, txt=f"Predicted Disease: {data['disease']}", ln=True, align="C")

    pdf.set_text_color(0)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Description: {data['description']}\n")
    pdf.ln(2)

    pdf.cell(0, 10, "Precautions:", ln=True)
    for p in data['precautions']:
        pdf.cell(0, 10, f"- {p}", ln=True)

    pdf.ln(2)
    pdf.cell(0, 10, "Medicines:", ln=True)
    for m in data['medicines']:
        pdf.cell(0, 10, f"- {m}", ln=True)

    pdf.ln(2)
    pdf.cell(0, 10, "Tests:", ln=True)
    for t in data['tests']:
        pdf.cell(0, 10, f"- {t}", ln=True)

    pdf.ln(2)
    pdf.cell(0, 10, "Symptoms and Severity:", ln=True)
    for s in data['symptom_details']:
        pdf.cell(0, 10, f"- {s['symptom']}: Severity {s['severity']}", ln=True)

    path = f"static/temp/{data['name'].replace(' ', '_')}_report.pdf"
    os.makedirs("static/temp", exist_ok=True)
    pdf.output(path)
    return path
