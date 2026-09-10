# CODSOFT ML Internship

Three ML tasks for the CodSoft virtual internship. Each lives in its own TASK* folder
and is fully self-contained.

## Environment
- Use `.venv/bin/python` (a local virtualenv on Python 3.14). There is no system pip.
- Run scripts from inside the task folder, e.g. `cd TASK4_SPAM_SMS_DETECTION && ../.venv/bin/python train.py`.

## Conventions
- Python 3.11+, scikit-learn, pandas, matplotlib/seaborn. No deep learning frameworks.
- Every task has: notebook.ipynb (exploration + narrative), train.py (trains and saves
  the model to models/), predict.py (CLI: loads model, predicts on new input).
- Save all trained artifacts with joblib into <TASK>/models/.
- Save every plot as PNG into <TASK>/results/ — these go in the README and the demo video.
- Set random_state=42 everywhere. Splits are stratified.
- Never commit datasets or .pkl files. They're gitignored.

## Evaluation rules
- All three datasets are imbalanced. Accuracy alone is NOT an acceptable metric.
- Always report: precision, recall, F1 per class, macro-F1, ROC-AUC, and a confusion matrix.
- Compare at least 3 algorithms per task in a results table before picking a winner.

## Style
- Explain what each step does in markdown cells — this is a learning project, the
  notebook should read like a walkthrough.
- Do not invent or synthesise data. If a data file is missing, stop and say so.
- Do not silently drop rows to make something work. Report what you dropped and why.
