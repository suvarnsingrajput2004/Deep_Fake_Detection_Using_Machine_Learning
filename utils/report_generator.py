import os


def generate_report(result):
    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', 'deepfake_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DeepFake Detection Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Status:              {result['status']}\n")
        f.write(f"Fake Score:          {result['fake_score']:.4f}\n")
        f.write(f"Confidence Score:    {result['confidence']:.4f} ({result['confidence']*100:.1f}%)\n")
        f.write(f"FFT Artifact Score:  {result.get('fft_artifact_score', 0):.4f}\n")
        f.write(f"Frames Analyzed:     {result.get('frames_analyzed', 0)}\n")
        f.write(f"Models Used:         {result.get('models_used', 1)}\n\n")

        f.write("Techniques Employed:\n")
        f.write("-" * 40 + "\n")
        f.write("- Multi-Model Ensemble (XceptionNet, MobileNetV2, ResNet50V2,\n")
        f.write("  EfficientNetB0, CNN-LSTM, GAN Discriminator, ViT)\n")
        f.write("- Weighted Ensemble Voting (by validation accuracy)\n")
        f.write("- Temperature Scaling Calibration\n")
        f.write("- MTCNN Face Detection\n")
        f.write("- FFT Frequency Domain Analysis\n")
        f.write("- Focal Loss Training\n")
        f.write("- Frame-by-Frame Aggregation\n")
    return report_path
