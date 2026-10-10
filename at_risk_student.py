import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

df = pd.read_csv("students.csv")                                   # 1. LOAD
print(df.describe().round(1)); print("\nMissing values:\n", df.isna().sum())
df = df.fillna(df.median(numeric_only=True))                       # 2. PREPROCESS (fill missing)
X, y = df.drop(columns="status"), df["status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3,   # 3. DECISION TREE (ID3/C4.5 style)
                               min_samples_leaf=5, random_state=42).fit(Xtr, ytr)

plt.figure(figsize=(16, 8))                                        # 4. TREE DIAGRAM
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree: At-Risk vs Safe"); plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")

pred = model.predict(Xte)                                          # 5. EVALUATION
ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=model.classes_), display_labels=model.classes_).plot(cmap="Blues")
plt.title("Confusion Matrix (test set)"); plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\nAccuracy:", round(accuracy_score(yte, pred), 3))
print(classification_report(yte, pred))                            # precision / recall / F1
print("IF-THEN rules:\n", export_text(model, feature_names=list(X.columns)))
print("Feature importance:\n", pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False).round(3))
