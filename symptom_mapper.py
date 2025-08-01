# symptom_mapper.py

SYMPTOM_SYNONYMS = {
    "tired": "fatigue",
    "feverish": "high_fever",
    "throwing up": "vomiting",
    "frequent urination": "polyuria",
    "nausea and headache": "nausea",
    "sore throat": "throat_irritation",
    "body pain": "joint_pain",
    "sweating": "excessive_sweating",
    "swollen glands": "swollen_lymph_nodes",
    "upset stomach": "stomach_pain",
    "chest tightness": "chest_pain",
    "difficulty breathing": "breathlessness",
    "runny nose": "continuous_sneezing",
    "headache": "headache",
    "cold": "chills",
    "light headed": "dizziness",
    "back pain": "back_pain",
    "coughing": "cough",
    "weight loss": "weight_loss",
    "high temperature": "high_fever",
    "dry mouth": "dry_mouth",
    "dizzy": "dizziness",
    "blurred vision": "blurred_and_distorted_vision",
    "low appetite": "loss_of_appetite",
    "burning when urinating": "burning_micturition",
    "dehydrated": "dehydration",
    "shortness of breath": "breathlessness",
    "bloated": "abdominal_pain",
    "itchy skin": "itching"
}

def map_symptom(user_input):
    """
    Map input symptom string to standardized symptom in dataset.
    """
    normalized = user_input.lower().strip()
    return SYMPTOM_SYNONYMS.get(normalized, normalized.replace(" ", "_"))
