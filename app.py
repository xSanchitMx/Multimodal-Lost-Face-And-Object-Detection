from flask import Flask, request, render_template, redirect, url_for
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = "data/new_faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
QUERIES_FILE = "data/new_queries.txt"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # --- If text query submitted ---
        if "text_query" in request.form and request.form["text_query"].strip():
            text_query = request.form["text_query"].strip()
            with open(QUERIES_FILE, "a") as f:
                f.write(f"txt:{text_query}\n")
            return redirect(url_for("index"))

        # --- If image uploaded ---
        if "image_query" in request.files:
            img = request.files["image_query"]
            if img.filename != "":
                # Save with timestamp
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_{img.filename}"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                img.save(save_path)

                # Add entry to queries file
                with open(QUERIES_FILE, "a") as f:
                    f.write(f"img:{save_path}\n")
            return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
