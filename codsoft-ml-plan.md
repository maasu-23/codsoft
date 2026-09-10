# CodSoft ML Internship — Build Plan (Claude Code)

## What the brief actually requires

- Complete **at least 3 of the 5 tasks**
- One GitHub repo, tasks named `CODSOFT_TASK<NO>`
- A **demo video** per task, posted on LinkedIn with `#codsoft` and tagging `@CODSOFT`
- Update LinkedIn profile
- Submission form comes later by email

## Which 3 tasks to pick

**Task 1 (Movie Genre Classification), Task 3 (Customer Churn), Task 4 (Spam SMS).**

Reasoning:
- Task 1 and Task 4 are the same pipeline (TF-IDF → linear classifier). Once you build one, the second is mostly a config change — you get two deliverables for ~1.3x the work.
- Task 3 is tabular, so your three tasks show *different* skills (NLP text + tabular ML), which reads better on a portfolio than three text classifiers.
- Task 2 (fraud) uses a ~1.5M-row dataset — slow to download and train in a hostel connection. Task 5 (handwriting RNN) needs GPU time and produces the weakest demo.

Swap Task 3 → Task 2 only if you specifically want a fraud-detection project on your resume.

## Datasets (Kaggle)

| Task | Kaggle slug | Files |
|---|---|---|
| 1 | `hijest/genre-classification-dataset-imdb` | `train_data.txt`, `test_data.txt`, `test_data_solution.txt` |
| 3 | `shivan118/churn-modeling-dataset` | `Churn Modelling.csv` |
| 4 | `uciml/sms-spam-collection-dataset` | `spam.csv` |

Setup once:

```bash
pip install kaggle
# Kaggle → Settings → API → Create New Token → saves kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d hijest/genre-classification-dataset-imdb -p TASK1/data --unzip
kaggle datasets download -d shivan118/churn-modeling-dataset -p TASK3/data --unzip
kaggle datasets download -d uciml/sms-spam-collection-dataset -p TASK4/data --unzip
```

## Repo layout

```
CODSOFT/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── .gitignore          # data/*.csv, data/*.txt, models/*.pkl
├── TASK1_MOVIE_GENRE_CLASSIFICATION/
│   ├── data/
│   ├── notebook.ipynb
│   ├── train.py
│   ├── predict.py
│   ├── models/
│   └── results/        # confusion matrix, class distribution PNGs
├── TASK3_CUSTOMER_CHURN_PREDICTION/
│   └── (same shape)
└── TASK4_SPAM_SMS_DETECTION/
    └── (same shape)
```

The brief's naming is ambiguous ("separate repository … name as CODSOFT_TASKNO"). One repo called `CODSOFT` with `TASK1/TASK3/TASK4` folders is what nearly every submission does and is fine. If you want to follow it literally instead, make three repos: `CODSOFT_TASK1`, `CODSOFT_TASK3`, `CODSOFT_TASK4`.

Keep `predict.py` in every task — a CLI that loads the saved model and predicts on new input. That's what makes the demo video watchable.

---

## Step 0 — CLAUDE.md

Create the repo, then drop this in as `CLAUDE.md` before you start. Claude Code reads it automatically every session.

```markdown
# CODSOFT ML Internship

Three ML tasks for the CodSoft virtual internship. Each lives in its own TASK* folder
and is fully self-contained.

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
```

---

## Step 1 — Task 4 first (Spam SMS)

Start here. It's the smallest dataset and fastest feedback loop, so you learn the workflow on the easy one.

**Prompt to paste into Claude Code:**

> Work in `TASK4_SPAM_SMS_DETECTION/`. The dataset is at `data/spam.csv`.
>
> Heads up on the file: it's `latin-1` encoded (utf-8 will fail), the real columns are `v1` (label: ham/spam) and `v2` (message text), and there are three junk unnamed columns to drop. Roughly 13% of rows are spam.
>
> Build:
> 1. `notebook.ipynb` — load and clean, EDA (class balance, message-length distribution by class, top tokens per class), then modelling.
> 2. TF-IDF features. Try both word n-grams (1,2) and char n-grams (3,5) and tell me which wins.
> 3. Compare MultinomialNB, LogisticRegression(class_weight='balanced'), and LinearSVC(class_weight='balanced') on a stratified 80/20 split. Put results in a comparison table.
> 4. Report precision/recall/F1 for the **spam** class specifically — recall matters more than accuracy here, and say why in a markdown cell.
> 5. Confusion matrix saved to `results/confusion_matrix.png`.
> 6. `train.py` that trains the winning pipeline and saves it to `models/spam_classifier.pkl`.
> 7. `predict.py` — CLI taking a message as an argument, printing the label plus confidence.
>
> Use a sklearn `Pipeline` so the vectoriser is saved with the model. Don't fit the vectoriser on the test split.

Then check it yourself:

```bash
python train.py
python predict.py "Congratulations! You've won a free iPhone, click here to claim"
python predict.py "hey are you coming to class tomorrow"
```

## Step 2 — Task 1 (Movie Genre)

**Prompt:**

> Work in `TASK1_MOVIE_GENRE_CLASSIFICATION/`. Data is in `data/`.
>
> File format — these are `:::`-delimited text files, not CSVs:
> - `train_data.txt`: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
> - `test_data.txt`: `ID ::: TITLE ::: DESCRIPTION` (no label)
> - `test_data_solution.txt`: `ID ::: TITLE ::: GENRE ::: DESCRIPTION` (the labels)
>
> Parse with `pd.read_csv(path, sep=' ::: ', engine='python', names=[...])`.
>
> Build:
> 1. `notebook.ipynb` — parse, EDA (there are ~27 genres and the distribution is badly skewed: drama and documentary dominate, several genres have very few examples — plot this and say what it means for the metrics).
> 2. TF-IDF on the plot description. Try with and without the title concatenated in.
> 3. Compare MultinomialNB, LogisticRegression, and LinearSVC — all with `class_weight='balanced'` where supported.
> 4. Evaluate on the real held-out test set using `test_data_solution.txt`, not just a validation split. Report **macro-F1** as the headline metric, plus accuracy, plus per-genre F1. Call out explicitly which genres the model fails on.
> 5. Normalised confusion matrix → `results/confusion_matrix.png`; genre distribution → `results/genre_distribution.png`.
> 6. `train.py` saving to `models/genre_classifier.pkl`, and `predict.py` taking a plot summary and printing the top 3 predicted genres with probabilities.
>
> Expect accuracy in the high 50s / low 60s — that's normal for this dataset with a linear model. Do not tune to chase a number; explain why the ceiling is low instead.

## Step 3 — Task 3 (Customer Churn)

**Prompt:**

> Work in `TASK3_CUSTOMER_CHURN_PREDICTION/`. Data at `data/Churn Modelling.csv` — 10,000 bank customers, target column `Exited` (1 = churned, ~20% of rows).
>
> Build:
> 1. `notebook.ipynb` — drop `RowNumber`, `CustomerId`, `Surname` (identifiers, no signal, and leaving them in leaks nothing useful). EDA: churn rate by Geography, Gender, Age bucket, NumOfProducts, IsActiveMember. Correlation heatmap for the numeric features.
> 2. Preprocessing in a `ColumnTransformer`: one-hot `Geography` and `Gender`, `StandardScaler` on numeric features (needed for LogReg, harmless for the trees).
> 3. Compare LogisticRegression, RandomForestClassifier, and GradientBoostingClassifier on a stratified 80/20 split. Comparison table with accuracy, precision, recall, F1, ROC-AUC.
> 4. Because the data is imbalanced, also show what happens with `class_weight='balanced'` and discuss the precision/recall tradeoff — a bank cares about catching churners (recall), and say that in the notebook.
> 5. Feature importances from the best tree model → `results/feature_importance.png`. ROC curves for all three on one axes → `results/roc_curves.png`. Confusion matrix for the winner.
> 6. `train.py` → `models/churn_model.pkl`. `predict.py` taking customer attributes as CLI flags and printing churn probability.
>
> Expect ~86-87% accuracy and ROC-AUC around 0.85-0.87 from the boosted model. If you get much higher, you have a leak — go find it.

## Step 4 — README

**Prompt:**

> Write the root `README.md`: what CodSoft is, the three tasks with the dataset link for each, repo structure, setup/install instructions, how to run each task, and a results summary table (task, best model, headline metric). Embed the PNGs from each `results/` folder. Then a short README inside each task folder covering approach, results table, and what you'd improve with more time.

---

## Claude Code working tips

- Run `claude` from the repo root so it picks up `CLAUDE.md`.
- Use **plan mode** (Shift+Tab twice) for each task prompt — read the plan before it writes anything.
- Commit after each task: `git add . && git commit -m "Task 4: spam SMS detection"`. Three clean commits beat one giant one.
- If it starts inventing numbers or fabricating a dataset because a file is missing, stop it — the `CLAUDE.md` rule covers this but check anyway.
- Ask it to run the training and paste the actual output rather than predicting metrics.

## Before the demo video

The video is where this gets checked. Go through each notebook once and make sure you can answer, unprompted:

- Why TF-IDF and not raw counts?
- Why is accuracy misleading on your dataset, and what did you use instead?
- What does your confusion matrix show is going wrong?
- Why did the winning algorithm beat the others?

If any answer isn't there, ask Claude Code to explain that section, then re-record. A 2-3 minute screen recording — problem, dataset, notebook scroll, live `predict.py` run — is enough.

## Order of operations

1. Repo + `CLAUDE.md` + `.gitignore` + Kaggle downloads — 30 min
2. Task 4 — 1 hour
3. Task 1 — 1.5 hours
4. Task 3 — 1.5 hours
5. READMEs + push — 30 min
6. Three demo videos + LinkedIn posts — 1 hour
