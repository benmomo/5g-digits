from flask import Flask, request, jsonify
from prometheus_client import Counter, generate_latest

app = Flask(__name__)

# Métrica muy simple para tener algo que ver en Prometheus/Grafana
REQUEST_COUNT = Counter(
    'mec_requests_total',
    'Total MEC requests',
    ['endpoint', 'method']
)

@app.route("/health", methods=["GET"])
def health():
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return jsonify({"status": "ok"}), 200

@app.route("/control", methods=["POST"])
def control():
    REQUEST_COUNT.labels(endpoint="/control", method="POST").inc()
    data = request.get_json() or {}
    cmd = data.get("cmd", "unknown")
    intersection = data.get("intersection_id", "unknown")
    return jsonify({
        "result": "accepted",
        "cmd": cmd,
        "intersection_id": intersection
    }), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    # endpoint para que Prometheus raspe métricas
    return generate_latest(), 200, {
        "Content-Type": "text/plain; charset=utf-8"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
