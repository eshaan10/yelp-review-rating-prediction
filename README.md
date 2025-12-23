# Yelp Review Rating Prediction

**Multi-class NLP classification** to predict Yelp star ratings (1–5) using only review text.
Final project for **UCI CS 178 (Machine Learning, Fall 2025)**.

---

## Overview

This project tackles real-world challenges in text classification:

* **Severe class imbalance** (≈44% of reviews are 5-star)
* **High-dimensional sparse text features**
* Careful **model validation, selection, and adaptation**

We evaluate multiple models and show why **TF-IDF + Logistic Regression** outperforms more complex alternatives on this task.

---

## Key Results

* **Best model:** TF-IDF + Logistic Regression
* **Accuracy:** ~63–67% (vs. 44% majority-class baseline)
* Strong generalization with minimal overfitting
* More complex models (Random Forests, SBERT) did **not** outperform simpler linear methods

---

## Methods

* **Text Representation:** TF-IDF
* **Validation:** Stratified 70 / 15 / 15 train–validation–test split
* **Models Explored:**

  * Dummy baseline
  * Logistic Regression (regularization tuning)
  * Naive Bayes (smoothing tuning)
  * Random Forest (TF-IDF)
  * SBERT embeddings + Linear SVM (extra)
* **Analysis:** Validation curves, underfitting vs. overfitting diagnosis

---

## Repository Structure

```
yelp-review-rating-prediction/
│
├── data/
│   └── data_goes_in_this_folder.txt
│
├── notebook/
│   ├── analysis.ipynb        # EDA & preprocessing
│   ├── master.ipynb          # Full modeling pipeline
│   └── fig1_star_distribution.png
│
├── README.md
```

---

## Dataset

Uses the **Yelp Open Dataset** (JSON format).
The dataset is **not included** due to size and licensing.

Expected structure after extraction:

```
data/yelp_dataset/
├── yelp_academic_dataset_review.json   # primary file used
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_user.json
├── yelp_academic_dataset_checkin.json
└── yelp_academic_dataset_tip.json
```

Only review text and star ratings are used for prediction.

---

## How to Run

1. Download the dataset: [https://business.yelp.com/data/resources/open-dataset/](https://business.yelp.com/data/resources/open-dataset/)
2. Extract into `data/yelp_dataset/`
3. Run:

   * `analysis.ipynb`
   * `master.ipynb`

---

## Tech Stack

* Python, NumPy, Pandas
* scikit-learn
* TF-IDF
* Sentence-BERT (extra)
* Matplotlib

---

## Takeaway

This project demonstrates that **well-regularized linear models with the right feature representation can outperform deeper or more complex models** on sparse NLP tasks—especially under data imbalance and runtime constraints.
