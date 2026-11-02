
import pandas as pd
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = []

for attendance in range(50, 101, 5):
    for internal in range(20, 101, 10):
        for assignment in range(20, 101, 10):
            for previous in range(20, 101, 10):
                average = (
                    attendance * 0.2 +
                    internal * 0.3 +
                    assignment * 0.2 +
                    previous * 0.3
                )
                result = 1 if average >= 60 else 0
                data.append([
                    attendance, internal,
                    assignment, previous, result
                ])

df = pd.DataFrame(data, columns=[
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
    "result"
])

df.to_csv("student_results.csv", index=False)

X = df[[
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "student_result_model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": float(accuracy)}, f, indent=4)

print("Model trained successfully!")
print("Accuracy:", accuracy)
