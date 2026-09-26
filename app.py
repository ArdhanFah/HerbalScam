import os
import uuid
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from config import Config
from services.ml_service import predict_leaf
from services.firestore_service import save_scan_history

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# ──────────────────────────────────────
# Page Routes
# ──────────────────────────────────────

@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/scan")
def scan():
    """Scan / upload page."""
    return render_template("scan.html")


@app.route("/result/<result_id>")
def result(result_id):
    """Result page — renders server-side result."""
    return render_template("result.html", result_id=result_id)


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


# ──────────────────────────────────────
# API Routes
# ──────────────────────────────────────

@app.route("/api/classify", methods=["POST"])
def api_classify():
    """
    Accepts an image upload, runs the classifier, and returns JSON.
    """
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Format file tidak didukung. Gunakan JPG, PNG, atau WebP."
        }), 400

    # Save the file with a unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)

    # Classify the image using ML Service
    result = predict_leaf(filepath)
    result["image_url"] = url_for("uploaded_file", filename=unique_filename, _external=True)

    # Optional: Save to Firestore if user_id is provided in request form/headers
    user_id = request.form.get("user_id") or request.headers.get("X-User-ID")
    if user_id:
        scan_id = save_scan_history(user_id, result)
        if scan_id:
            result["scan_id"] = scan_id

    return jsonify(result)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ──────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    return render_template("base.html", error_code=404, error_message="Halaman tidak ditemukan"), 404


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "Ukuran file terlalu besar. Maksimal 16 MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return render_template("base.html", error_code=500, error_message="Terjadi kesalahan server"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
