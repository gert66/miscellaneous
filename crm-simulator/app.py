"""
Time-to-CRM Simulator — Flask entry point.
Run:  python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from simulation import run_simulation, params_from_dict

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/simulate", methods=["POST"])
def simulate():
    body = request.get_json(force=True, silent=True) or {}
    try:
        params = params_from_dict(body)
        result = run_simulation(params)
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    print("=" * 55)
    print("  Time-to-CRM Simulator")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host="127.0.0.1", port=5000)
