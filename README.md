# Task 03 — Decision Tree Classifier
### SkillCraft Technology Internship

---

## Objective

Build a decision tree classifier to predict whether a customer will subscribe to a term deposit, based on demographic and behavioral data from the Bank Marketing dataset (UCI Machine Learning Repository).

---

## Dataset

**Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**File used:** `bank-additional-full.csv`

| Property | Value |
|---|---|
| Total rows | 41,188 |
| Total features | 20 |
| Target column | `y` (yes / no) |
| Separator | semicolon (`;`) |
| Missing values | None (uses `"unknown"` as placeholder) |

---

## Project Structure

```
task-03-decision-tree/
│
├── bank-additional-full.csv          # Raw dataset
├── bank_cleaned.csv                  # Encoded dataset (used for training)
├── bank_cleaned_original.csv         # Human-readable cleaned dataset
│
├── decision_tree.ipynb               # Main Jupyter Notebook (all steps)
│
├── decision_tree.png                 # Visualized decision tree
├── feature_importances.png           # Feature importance bar chart
├── feature_importances_horizontal.png
│
└── README.md                         # This file
```

---

## Steps Followed

### Step 1 — Load the dataset
Loaded `bank-additional-full.csv` using `pandas.read_csv()` with `sep=';'`.

### Step 2 — Exploratory Data Analysis (EDA)
- Checked shape, data types, and column names
- Verified missing values (none found)
- Identified `"unknown"` string placeholders in categorical columns
- Analyzed target class distribution (imbalanced: ~89% No, ~11% Yes)
- Visualized age distribution and correlation heatmap

### Step 3 — Preprocessing
- Applied `LabelEncoder` to all categorical (object) columns
- Encoded target column `y` → 0 (No), 1 (Yes)
- Saved cleaned encoded data as `bank_cleaned.csv`

### Step 4 — Train / Test Split
- Split data: 80% training, 20% testing
- Used `stratify=y` to preserve class ratio in both splits
- Training set: ~32,950 rows | Test set: ~8,238 rows

### Step 5 — Model Training
- Trained `DecisionTreeClassifier` with `max_depth=5`, `criterion='gini'`
- Compared accuracy across depths 3–10 to identify optimal depth
- Selected `max_depth=5` as the best balance between accuracy and overfitting

### Step 6 — Evaluation
- Calculated accuracy, precision, recall, and F1-score
- Generated confusion matrix heatmap
- Plotted ROC curve and calculated AUC score

### Step 7 — Visualization
- Plotted full decision tree using `plot_tree()`
- Generated feature importance bar charts

---

## Results

| Metric | Value |
|---|---|
| Training Accuracy | ~91% |
| Test Accuracy | ~90% |
| AUC Score | ~0.80 |

---

## Key Findings

- **Call duration (`duration`)** was the single most important feature — longer calls strongly indicate customer interest in subscribing.
- **Economic indicators** like `euribor3m` and `nr.employed` were the next most influential features.
- The dataset is **imbalanced** (9:1 ratio), so accuracy alone is misleading — recall and F1-score for the "Yes" class are more meaningful metrics.
- Beyond `max_depth=5`, training accuracy keeps climbing while test accuracy plateaus — a clear sign of overfitting.

---

## Libraries Used

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone or download this folder
2. Place `bank-additional-full.csv` in the same directory
3. Open `decision_tree.ipynb` in Jupyter Notebook or VS Code
4. Run all cells top to bottom

---

## Author

**Intern:** Aish  
**Organization:** SkillCraft Technology  
**Task:** Task 03 — Decision Tree Classifier  
