# Healthcare Assistant Web App

This web-based healthcare assistant predicts the most likely disease based on user symptoms and provides the description, precautions, suggested medicines, and recommended medical tests.

## Features

- Input symptoms via dropdowns and/or text
- Synonym mapping for symptoms
- Disease prediction using MLP model
- Symptom similarity mode (alternative logic)
- Text-to-speech output of diagnosis
- Download result as PDF
- Voice input support
- Dark/Light theme toggle

## Technologies Used

- Python (Flask)
- Scikit-learn (MLPClassifier)
- HTML, CSS, Bootstrap
- JavaScript (for voice and theme functionality)
- xhtml2pdf (PDF generation)

## Project Structure

healthcare_app/
├── app.py
├── symptom_mapper.py
├── templates/
├── static/
├── dataset/
├── data/
├── requirements.txt
└── README.md

r
Copy
Edit

## Setup Instructions

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/healthcare-assistant.git
    cd healthcare-assistant
    ```

2. Create a virtual environment and install dependencies:
    ```
    python -m venv venv
    venv\Scripts\activate     # Windows
    pip install -r requirements.txt
    ```

3. Run the Flask app:
    ```
    python app.py
    ```

4. Open your browser and go to:
    ```
    http://127.0.0.1:5000
    ```

## Notes

- You can modify the MLP model or retrain it using the dataset.
- Voice input uses the Web Speech API (Chrome recommended).

## License

This project is licensed for educational and demonstration purposes.
