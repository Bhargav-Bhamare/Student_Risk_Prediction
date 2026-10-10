# Student At-Risk Prediction (Streamlit)

A simple Streamlit interface for the existing Decision Tree classification project.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Create a public GitHub repository and upload all files in this folder.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit will install the packages listed in `requirements.txt`.

The app preserves the original median imputation, stratified 70/30 split (`random_state=42`), and Decision Tree parameters (`criterion='entropy'`, `max_depth=3`, `min_samples_leaf=5`, `random_state=42`). It adds a form to test predictions.
