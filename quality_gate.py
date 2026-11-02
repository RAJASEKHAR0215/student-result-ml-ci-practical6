
import json
import sys

MINIMUM_ACCURACY = 0.85

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print("Model Accuracy:", accuracy)
print("Required Accuracy:", MINIMUM_ACCURACY)

if accuracy >= MINIMUM_ACCURACY:
    print("QUALITY GATE PASSED!")
else:
    print("QUALITY GATE FAILED!")
    sys.exit(1)
