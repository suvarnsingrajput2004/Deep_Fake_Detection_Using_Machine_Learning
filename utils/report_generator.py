import os

def generate_report(result):
    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', 'deepfake_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DeepFake Detection Report\n")
        f.write("=========================\n")
        f.write(f"Status: {result['status']}\n")
        f.write(f"Confidence Score: {result['average_score']:.4f}\n\n")
        f.write("Techniques employed:\n")
        f.write("- CNN Facial Analysis\n- Frame-by-Frame Temporal Consistency\n- Dlib Face Detection\n")
    return report_path
