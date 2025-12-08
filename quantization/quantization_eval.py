# This script will generate:
'''
accuracy_comparison.png
latency_comparison.png
model_size_comparison.png
quality_vs_latency.png
'''




import os
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Poisson


# CONFIG — CHANGE THESE PATHS
FP32_PATH = "models/fp32_student/"
PTQ_PATH  = "models/ptq_int8/"
QAT_PATH  = "models/qat_int8/"

CALIB_DATA_PATH = "data/calibration.npy"
BATCH_SIZE = 512



# Helper Functions
def load_model(path):
    print(f"Loading model from {path} ...")
    return tf.keras.models.load_model(path)


def model_size_mb(path):
    """Return size of SavedModel directory in MB."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def measure_latency(model, sample):
    """Returns inference latency in milliseconds per sample."""
    # warmup
    _ = model(sample)

    start = time.time()
    _ = model(sample)
    end = time.time()

    return (end - start) * 1000  # ms


def compute_accuracy(model, x, y):
    preds = model.predict(x, batch_size=BATCH_SIZE)
    metric = Poisson()
    metric.update_state(y, preds)
    return float(metric.result().numpy())



# Main Evaluation
def main():

    # Load data
    data = np.load(CALIB_DATA_PATH, allow_pickle=True).item()
    x_test, y_test = data["x"], data["y"]
    sample_input = tf.convert_to_tensor(x_test[:1])

    # Load models
    m_fp32 = load_model(FP32_PATH)
    m_ptq  = load_model(PTQ_PATH)
    m_qat  = load_model(QAT_PATH)

    # 1. Accuracy
    print("\nEvaluating Accuracy...")
    acc_fp32 = compute_accuracy(m_fp32, x_test, y_test)
    acc_ptq  = compute_accuracy(m_ptq,  x_test, y_test)
    acc_qat  = compute_accuracy(m_qat,  x_test, y_test)
    print("-> Accuracy Done")

    # 2. Latency
    print("\nMeasuring Latency...")
    lat_fp32 = measure_latency(m_fp32, sample_input)
    lat_ptq  = measure_latency(m_ptq,  sample_input)
    lat_qat  = measure_latency(m_qat,  sample_input)
    print("-> Latency Done")

    # 3. Model size
    print("\nMeasuring Model Size...")
    size_fp32 = model_size_mb(FP32_PATH)
    size_ptq  = model_size_mb(PTQ_PATH)
    size_qat  = model_size_mb(QAT_PATH)
    print("-> Model Size Done")


    # PLOTS
    # 1. Accuracy Chart
    plt.figure(figsize=(7,5))
    plt.bar(["FP32", "PTQ", "QAT"], [acc_fp32, acc_ptq, acc_qat], color=["blue","green","orange"])
    plt.ylabel("Poisson Log Loss (lower = better)")
    plt.title("Accuracy Comparison")
    plt.savefig("accuracy_comparison.png")
    plt.close()

    # 2. Latency Chart
    plt.figure(figsize=(7,5))
    plt.bar(["FP32", "PTQ", "QAT"], [lat_fp32, lat_ptq, lat_qat], color=["blue","green","orange"])
    plt.ylabel("Latency (ms per inference)")
    plt.title("Latency Comparison")
    plt.savefig("latency_comparison.png")
    plt.close()

    # 3. Model Size Chart
    plt.figure(figsize=(7,5))
    plt.bar(["FP32", "PTQ", "QAT"], [size_fp32, size_ptq, size_qat], color=["blue","green","orange"])
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size Comparison")
    plt.savefig("model_size_comparison.png")
    plt.close()

    # 4. Quality vs Latency (Pareto Curve)
    plt.figure(figsize=(7,5))
    plt.scatter([lat_fp32, lat_ptq, lat_qat], [acc_fp32, acc_ptq, acc_qat],
                s=[120,120,120], c=["blue","green","orange"], label="models")
    for x,y,name in [(lat_fp32,acc_fp32,"FP32"),(lat_ptq,acc_ptq,"PTQ"),(lat_qat,acc_qat,"QAT")]:
        plt.text(x, y, name)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Poisson Log Loss")
    plt.title("Quality vs Latency Trade-off (Pareto Curve)")
    plt.savefig("quality_vs_latency.png")
    plt.close()

    print("\nAll graphs saved:")
    print("  - accuracy_comparison.png")
    print("  - latency_comparison.png")
    print("  - model_size_comparison.png")
    print("  - quality_vs_latency.png")
    print("\nDone ✔")


if __name__ == "__main__":
    main()
