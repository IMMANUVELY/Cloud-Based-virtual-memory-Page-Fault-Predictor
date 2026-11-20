
"""
Cloud Based Virtual Memory Page Fault Predictor
NO AI VERSION
Computer Architecture Project

Run:
    python main_no_ai.py
    python main_no_ai.py --serve
    python main_no_ai.py --client
"""

import sys
import random
from collections import OrderedDict, deque

import numpy as np
import matplotlib.pyplot as plt

WINDOW = 16
THRESHOLD = 0.5


# STEP 1: Generate memory page requests
def generate_trace(n=300, seed=7):
    random.seed(seed)
    trace = []
    base = 0

    while len(trace) < n:
        kind = random.choice(["loop", "seq", "hot", "random"])

        if kind == "loop":
            pages = [
                base + i for i in range(random.randint(3, 8))
            ]
            for _ in range(random.randint(5, 15)):
                trace += pages

        elif kind == "seq":
            start = random.randint(0, 200)
            trace += list(
                range(start, start + random.randint(20, 60))
            )

        elif kind == "hot":
            hot = random.sample(range(40), 6)
            trace += [
                random.choice(hot)
                for _ in range(random.randint(40, 100))
            ]

        else:
            trace += [
                random.randint(0, 200)
                for _ in range(random.randint(20, 50))
            ]

        base = random.randint(0, 150)

    return trace[:n]


# STEP 2: RAM simulation using LRU
def simulate_lru(trace, frames=16):
    ram = OrderedDict()
    faults = []

    for page in trace:

        if page in ram:
            ram.move_to_end(page)
            faults.append(0)

        else:
            faults.append(1)

            if len(ram) >= frames:
                ram.popitem(last=False)

            ram[page] = True

    return faults


# STEP 3: Extract 8 features
def features_from_history(pages, faults):

    pages = list(pages)
    faults = list(faults)

    if not pages or not faults:
        raise ValueError("History cannot be empty")

    differences = [
        pages[i] - pages[i - 1]
        for i in range(1, len(pages))
    ]

    last_delta = differences[-1] if differences else 0

    stride_same = int(
        len(differences) >= 2
        and differences[-1] == differences[-2]
    )

    return [
        pages[-1],
        last_delta,
        stride_same,
        len(set(pages)),
        sum(faults) / len(faults),
        faults[-1],
        sum(faults[-4:]),
        int(
            np.mean(np.abs(differences))
            if differences else 0
        )
    ]


# STEP 4: Rule-based predictor (NO AI)
def rule_probability(features):

    stride_same = features[2]
    last_fault = features[5]
    faults_last4 = features[6]

    # Rule 1: Fault burst
    if faults_last4 >= 3:
        probability = (
            0.70
            + 0.10 * stride_same
            + 0.10 * last_fault
        )

    # Rule 2: Last access was a fault
    elif last_fault:
        probability = 0.55 + 0.10 * faults_last4

    # Rule 3: Last access was a hit
    else:
        probability = 0.02 + 0.15 * faults_last4

    return float(min(1.0, max(0.0, probability)))


# STEP 5: Make prediction
def predict(features, threshold=THRESHOLD):

    probability = rule_probability(features)

    action = (
        "PREFETCH"
        if probability > threshold
        else "NORMAL"
    )

    return {
        "fault_probability": probability,
        "action": action
    }


# STEP 6: Calculate evaluation metrics
def calculate_metrics(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    total = len(actual)

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# STEP 7: Evaluate predictor
def evaluate():

    trace = generate_trace(n=20000, seed=1)
    faults = simulate_lru(trace)

    actual = []
    predicted = []

    for t in range(WINDOW - 1, len(trace) - 1):

        features = features_from_history(
            trace[t - WINDOW + 1:t + 1],
            faults[t - WINDOW + 1:t + 1]
        )

        result = predict(features)

        actual.append(faults[t + 1])
        predicted.append(
            int(result["fault_probability"] > THRESHOLD)
        )

    metrics = calculate_metrics(actual, predicted)

    print("\nCLOUD BASED VIRTUAL MEMORY PAGE FAULT PREDICTOR")
    print("--------------- NO AI VERSION ------------------")

    print("Total memory accesses:", len(trace))
    print("Total page faults:", sum(faults))
    print("Page fault rate:", f"{np.mean(faults):.2%}")

    print("\nPREDICTION PERFORMANCE")

    print("Accuracy :", f"{metrics['accuracy']:.3f}")
    print("Precision:", f"{metrics['precision']:.3f}")
    print("Recall   :", f"{metrics['recall']:.3f}")
    print("F1 Score :", f"{metrics['f1_score']:.3f}")

    print("\nCONFUSION MATRIX")
    print([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ])

    plt.figure(figsize=(10, 4))
    plt.plot(actual[:200], label="Actual Fault")
    plt.plot(
        predicted[:200],
        "--",
        label="Predicted Fault"
    )
    plt.title("Actual vs Predicted Page Fault")
    plt.xlabel("Memory Access")
    plt.ylabel("Page Fault")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    plt.close()

    print("\nGraph saved: prediction_plot.png")


# STEP 8: Flask prediction API
def serve():

    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "message": "Page Fault Predictor API is running"
        })

    @app.route("/predict", methods=["POST"])
    def api():

        data = request.get_json(silent=True) or {}
        features = data.get("features")

        if not isinstance(features, list) or len(features) != 8:
            return jsonify({
                "error": "Exactly 8 features are required"
            }), 400

        try:
            features = [float(x) for x in features]
            result = predict(features)
            return jsonify(result)

        except (ValueError, TypeError):
            return jsonify({
                "error": "Invalid feature values"
            }), 400

    app.run(host="127.0.0.1", port=5000, debug=False)


# STEP 9: Simulated OS client
def client():

    import requests

    trace = generate_trace(n=300, seed=7)
    faults = simulate_lru(trace)

    page_window = deque(maxlen=WINDOW)
    fault_window = deque(maxlen=WINDOW)

    prefetch_count = 0

    for t, (page, fault) in enumerate(zip(trace, faults)):

        page_window.append(page)
        fault_window.append(fault)

        if len(page_window) == WINDOW:

            features = features_from_history(
                page_window,
                fault_window
            )

            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"features": features},
                timeout=5
            )

            result = response.json()

            if result["action"] == "PREFETCH":
                prefetch_count += 1

            if t < WINDOW + 5:
                print(t, result)

    print("\nPrefetch actions triggered:", prefetch_count)


# MAIN PROGRAM
if __name__ == "__main__":

    if "--serve" in sys.argv:
        serve()

    elif "--client" in sys.argv:
        client()

    else:
        evaluate()