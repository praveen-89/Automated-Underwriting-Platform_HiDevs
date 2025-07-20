import os
import pytesseract
import cv2
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

REPORT_FILE = "../data/reports.txt"
IMAGE_DIR = "../images/"

def extract_text_entities(text):
    nlp = pipeline("ner", grouped_entities=True)
    entities = nlp(text)
    return entities

def load_reports(file_path):
    with open(file_path, "r") as f:
        return f.read().split("\n---\n")

def analyze_images(image_dir):
    results = []
    for file in os.listdir(image_dir):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            text = pytesseract.image_to_string(img)

            # Optional risk scoring
            score = 0
            reasons = []
            if "crack" in text.lower():
                score += 3
                reasons.append("Cracks detected")
            if "leak" in text.lower():
                score += 2
                reasons.append("Leakage mentioned")

            results.append({
                "Image File": file,
                "OCR Text": text,
                "Image Risk Score": score,
                "Image Risk Factors": ", ".join(reasons)
            })
    return results


import re

# Extract structured data from a report
def parse_report(text):
    def extract(pattern, default="N/A"):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default

    return {
        "Property ID": extract(r"Property ID:\s*(\d+)"),
        "Location": extract(r"Location:\s*(.+)"),
        "Year Built": extract(r"Year Built:\s*(\d{4})"),
        "Condition": extract(r"Condition:\s*(.+)"),
        "Estimated Value": extract(r"Estimated Value:\s*\$?([\d,]+)"),
        "Hazards": extract(r"Hazards:\s*(.+)")
    }

# Simple rule-based risk scoring
def score_risk(report_dict):
    risk_score = 0
    reasons = []

    condition = report_dict["Condition"].lower()
    hazards = report_dict["Hazards"].lower()

    if "water damage" in condition:
        risk_score += 3
        reasons.append("Water damage")
    if "flood" in hazards:
        risk_score += 4
        reasons.append("Flood risk")
    if int(report_dict["Year Built"]) < 2000:
        risk_score += 2
        reasons.append("Old construction")
    
    return risk_score, ", ".join(reasons)

def classify_risk(score):
    if score >= 7:
        return "High", "Reject"
    elif score >= 4:
        return "Medium", "Review"
    else:
        return "Low", "Approve"


# imports
# utility functions: extract_text_entities, load_reports, analyze_images
# NEW functions: parse_report, score_risk ✅

if __name__ == "__main__":
    print("=== Analyzing Reports ===")
    reports = load_reports(REPORT_FILE)

    results = []
    for report in reports:
        parsed = parse_report(report)
        score, reason = score_risk(parsed)
        parsed["Risk Score"] = score
        parsed["Risk Factors"] = reason
        results.append(parsed)

    print("\n=== Analyzing Images ===")
    image_results = analyze_images(IMAGE_DIR)
    image_df = pd.DataFrame(image_results)

    combined = []
    for r in results:
        prop_id = r["Property ID"]
        matching_row = image_df[image_df["Image File"].str.contains(prop_id)].squeeze()

        if not matching_row.empty:
            r["Image File"] = matching_row["Image File"]
            r["Image OCR Text"] = matching_row["OCR Text"]
            r["Image Risk Score"] = matching_row["Image Risk Score"]
            r["Image Risk Factors"] = matching_row["Image Risk Factors"]
            r["Total Risk Score"] = r["Risk Score"] + matching_row["Image Risk Score"]
        else:
            r["Image File"] = "N/A"
            r["Image OCR Text"] = ""
            r["Image Risk Score"] = 0
            r["Image Risk Factors"] = ""
            r["Total Risk Score"] = r["Risk Score"]

        # 🔥 Risk Classification & Decision
        risk_level, decision = classify_risk(r["Total Risk Score"])
        r["Risk Level"] = risk_level
        r["Underwriting Decision"] = decision

        combined.append(r)

    final_df = pd.DataFrame(combined)
    final_df.to_csv("../data/final_underwriting_report.csv", index=False)
    print("\n✅ Final underwriting report saved to data/final_underwriting_report.csv")
    print(final_df)



