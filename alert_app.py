from flask import Flask, render_template, send_from_directory, jsonify
import os
import json

app = Flask(__name__)

ALERTS_FILE = "alerts/alerts.json"
ALERTS_DIR = "alerts"

# Route to serve alert images
@app.route("/alerts/<path:filename>")
def serve_alert_image(filename):
    return send_from_directory(ALERTS_DIR, filename)

# API to fetch alerts as JSON (for AJAX/refresh)
@app.route("/api/alerts")
def get_alerts():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
    else:
        alerts = []
    return jsonify(alerts)

# Main page
@app.route("/")
def index():
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
    else:
        alerts = []
    return render_template("alerts.html", alerts=alerts)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
