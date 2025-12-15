# src/gates.py
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class GateThresholds:
    min_accuracy: float = 0.85
    max_latency_ms: float = 25.0
    max_model_size_mb: float = 5.0
    max_validation_fail_rate: float = 0.02  # 2%

def evaluate_gates(kpis: Dict[str, float], thresholds: GateThresholds) -> Tuple[bool, Dict[str, Any]]:
    checks = {}

    checks["accuracy"] = {
        "value": kpis.get("accuracy", 0.0),
        "threshold": thresholds.min_accuracy,
        "pass": kpis.get("accuracy", 0.0) >= thresholds.min_accuracy,
        "direction": "higher_is_better"
    }
    checks["latency_ms"] = {
        "value": kpis.get("latency_ms", 1e9),
        "threshold": thresholds.max_latency_ms,
        "pass": kpis.get("latency_ms", 1e9) <= thresholds.max_latency_ms,
        "direction": "lower_is_better"
    }
    checks["model_size_mb"] = {
        "value": kpis.get("model_size_mb", 1e9),
        "threshold": thresholds.max_model_size_mb,
        "pass": kpis.get("model_size_mb", 1e9) <= thresholds.max_model_size_mb,
        "direction": "lower_is_better"
    }
    checks["validation_fail_rate"] = {
        "value": kpis.get("validation_fail_rate", 1.0),
        "threshold": thresholds.max_validation_fail_rate,
        "pass": kpis.get("validation_fail_rate", 1.0) <= thresholds.max_validation_fail_rate,
        "direction": "lower_is_better"
    }

    overall_pass = all(v["pass"] for v in checks.values())
    details = {
        "overall_pass": overall_pass,
        "checks": checks
    }
    return overall_pass, details
