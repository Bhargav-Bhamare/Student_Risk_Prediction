"""Creates students.csv and students.arff (for WEKA) - sample data for the project.
Replace with real/your own data if available (same column names)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
n = 250
df = pd.DataFrame({
    "attendance":   np.clip(rng.normal(75, 15, n), 30, 100).round(1),
    "midterm":      np.clip(rng.normal(55, 18, n), 5, 100).round(1),
    "assignment_rate": np.clip(rng.normal(70, 22, n), 0, 100).round(1),
    "lab_quiz":     np.clip(rng.normal(60, 20, n), 0, 100).round(1),
})
# hidden "risk score": low attendance / marks -> more likely At-Risk (+ some noise)
score = (0.35*(75-df.attendance) + 0.40*(50-df.midterm) + 0.15*(60-df.assignment_rate)
         + 0.10*(50-df.lab_quiz) + rng.normal(0, 1.5, n))
df["status"] = np.where(score > 2, "At-Risk", "Safe")
df.to_csv("students.csv", index=False)
with open("students.arff", "w") as f:                       # ARFF file for WEKA
    f.write("@relation student_risk\n\n")
    for c in df.columns[:-1]: f.write(f"@attribute {c} numeric\n")
    f.write("@attribute status {At-Risk,Safe}\n\n@data\n")
    for _, r in df.iterrows(): f.write(",".join(str(v) for v in r.values) + "\n")
print(df.status.value_counts())
