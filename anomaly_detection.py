
import luminol

def get_anomalies(error):
    detector = luminol.anomaly_detector.AnomalyDetector(error)
    anomalies = detector.get_anomalies()
    return anomalies