import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

st.set_page_config(page_title="Student At-Risk Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student At-Risk Prediction")
st.caption("Supervised machine learning with a Decision Tree")
st.write("The model classifies student records as **At-Risk** or **Safe** using academic indicators.")

@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load students.csv: {exc}")
    st.stop()

required = ["attendance", "midterm", "assignment_rate", "lab_quiz", "status"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("students.csv is missing required columns: " + ", ".join(missing))
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Student records", len(df))
c2.metric("At-Risk records", int((df["status"] == "At-Risk").sum()))
c3.metric("Safe records", int((df["status"] == "Safe").sum()))
st.dataframe(df.head(10), use_container_width=True, hide_index=True)

# Preserve the original script's preprocessing, split, and model settings.
df = df.fillna(df.median(numeric_only=True))
X, y = df.drop(columns="status"), df["status"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, min_samples_leaf=5, random_state=42
).fit(Xtr, ytr)
pred = model.predict(Xte)
accuracy = accuracy_score(yte, pred)
report = classification_report(yte, pred, output_dict=True, zero_division=0)

st.subheader("Model performance")
a, b, c = st.columns(3)
a.metric("Test accuracy", f"{accuracy:.1%}")
b.metric("Training records", len(Xtr))
c.metric("Test records", len(Xte))

report_rows = []
for label in model.classes_:
    report_rows.append({
        "Class": label,
        "Precision": round(report[label]["precision"], 3),
        "Recall": round(report[label]["recall"], 3),
        "F1-score": round(report[label]["f1-score"], 3),
        "Support": int(report[label]["support"]),
    })
st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

cm = confusion_matrix(yte, pred, labels=model.classes_)
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(
    ax=ax_cm, cmap="Blues", colorbar=False
)
ax_cm.set_title("Confusion Matrix (test set)")
st.pyplot(fig_cm, use_container_width=True)
plt.close(fig_cm)

st.subheader("Decision Tree")
fig_tree, ax_tree = plt.subplots(figsize=(15, 7))
plot_tree(
    model, feature_names=list(X.columns), class_names=[str(c) for c in model.classes_],
    filled=True, rounded=True, fontsize=9, ax=ax_tree
)
ax_tree.set_title("Decision Tree: At-Risk vs Safe")
st.pyplot(fig_tree, use_container_width=True)
plt.close(fig_tree)

st.subheader("Feature importance")
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)
st.dataframe(importance.round(3), use_container_width=True, hide_index=True)

with st.expander("View model decision rules"):
    st.code(export_text(model, feature_names=list(X.columns)), language="text")

st.subheader("Try a prediction")
st.write("Enter academic indicators below. The result is a model prediction for demonstration, not a final judgement about a student.")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    attendance = c1.slider("Attendance (%)", 0.0, 100.0, 75.0, 0.5)
    midterm = c2.slider("Midterm marks", 0.0, 100.0, 55.0, 0.5)
    assignment_rate = c1.slider("Assignment rate (%)", 0.0, 100.0, 70.0, 0.5)
    lab_quiz = c2.slider("Lab / quiz score", 0.0, 100.0, 60.0, 0.5)
    submitted = st.form_submit_button("Predict student status")

if submitted:
    new_record = pd.DataFrame([{
        "attendance": attendance,
        "midterm": midterm,
        "assignment_rate": assignment_rate,
        "lab_quiz": lab_quiz,
    }])[list(X.columns)]
    result = model.predict(new_record)[0]
    if result == "At-Risk":
        st.warning("Prediction: At-Risk. Consider reviewing this student's situation and offering appropriate support.")
    else:
        st.success("Prediction: Safe. Continue normal academic monitoring.")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(new_record)[0]
        st.write("Model class probabilities:")
        st.dataframe(pd.DataFrame({"Class": model.classes_, "Probability": probs}).round(3), hide_index=True)

st.caption("For educational demonstration only. Predictions can be wrong and should not be used as the sole basis for academic decisions.")
