# KRAFT: KuaiRec Recommendation Algorithm for Feed Tailoring

**Objective:**
Develop a recommender system to suggest short videos to users based on preferences, interaction histories, and video content using the KuaiRec dataset. The goal is to create a personalized and scalable recommendation engine.

---

## 1. Introduction

This project implements a multi-stage hybrid recommender system for short videos, leveraging the KuaiRec dataset. The system provides personalized video suggestions by first generating a broad set of candidate videos for each user, then re-ranking these candidates using a model that incorporates rich user, item, and interaction features.

The KuaiRec dataset's `small_matrix` is described as "fully observed," meaning nearly every user has a `watch_ratio` for every item. Thus, the task is primarily one of **ranking** items based on predicted engagement rather than inferring missing interactions.

---

## 2. Project Setup

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

---

## 3. Methodology

The recommender system uses a two-stage architecture to balance scalability and prediction accuracy:

### 3.1 Data Preparation and Feature Engineering

Detailed in `notebooks/1_Data_Preparation.ipynb`, this phase includes:

* **Loading Raw Data:**
  Ingests `big_matrix.csv`, `small_matrix.csv`, `user_features.csv`, `item_categories.csv`, and `item_daily_features.csv`.

* **Initial Cleaning:**
  Handles missing values, data types, and parses complex fields (e.g., item tags).

* **Feature Creation:**

  * *Temporal Features:* Extract `interaction_hour` and `interaction_day_of_week`.
  * *User Features:* Includes `user_active_degree`, `fans_user_num`, `register_days`, `is_lowactive_period`, `is_live_streamer`, `is_video_author`, and `onehot_feat*`.
  * *Static Item Features:* Extracts `num_item_tags` from `item_categories.csv`.
  * *Dynamic Item Features:* Derives `video_age_days`, engagement ratios like `daily_play_per_show_ratio`, and raw stats like `video_duration_daily`.
  * *Duplicate Handling:* Removes duplicate entries in `item_daily_features` based on `(video_id, date)`.

* **Data Splitting & Saving:**

  * `big_matrix` is split chronologically: 80% train, 20% test.
  * Feature-engineered `small_matrix` saved separately.
  * All outputs saved in Parquet format.

### 3.2 Stage 1: Candidate Generation (ALS)

* **Model:**
  ALS from the `implicit` library (matrix factorization for implicit feedback).

* **Training Data:**
  `big_matrix` with clipped `watch_ratio > 0` used as confidence.

* **Output:**
  List of candidate item IDs per user.

* **Reference:**
  `notebooks/2_Model_Training.ipynb`

### 3.3 Stage 2: Ranking (LightGBM)

* **Model:**
  LightGBM (gradient boosting) for regression.

* **Task:**
  Predict `watch_ratio` for `(user, item)` pairs using engineered features.

* **Training Data:**
  80% training split of `big_matrix`.

* **Loss Function:**
  L1 loss (MAE), robust to outliers.

* **Output:**
  Predicted `watch_ratio` used to re-rank ALS candidates.

* **Reference:**
  `notebooks/2_Model_Training.ipynb`

---

## 4. Experiments and Evaluation

Detailed in `notebooks/3_Model_Evaluation.ipynb`.

### 4.1 Evaluation on `small_matrix` (Dense Subset)

* **Purpose:**
  Evaluate LightGBM on known users/items from training data.

* **Methodology:**

  * Predict `watch_ratio` for all items per user in `small_matrix`.
  * Rank items by predicted scores.

* **Metrics:**

  * *Pointwise:* RMSE, MAE
  * *Ranking:* Precision\@k, Recall\@k, nDCG\@k (k = 5, 10, 20, 50, 100, 250)

### 4.2 Evaluation on `big_matrix` Holdout Set

* **Purpose:**
  Evaluate on unseen interactions from test split.

* **Methodology:**

  * Use 20% test split.
  * Predict and rank only items the user interacted with.

* **Metrics:**
  Ranking metrics as above.

---

## 5. Results

### 5.1 `small_matrix` Evaluation Results

* **Pointwise Metrics:**

  * RMSE: `1.3214`
  * MAE: `0.3473`

* **Ranking Metrics (Relevance: `watch_ratio > 1.0`):**

| Metric        | @5     | @10    | @20    | @50    | @100   | @250   |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Avg Precision | 0.9137 | 0.9117 | 0.9019 | 0.8571 | 0.8234 | 0.7725 |
| Avg Recall    | 0.0048 | 0.0095 | 0.0188 | 0.0443 | 0.0848 | 0.1973 |
| Avg NDCG      | 0.9146 | 0.9128 | 0.9057 | 0.8706 | 0.8396 | 0.7904 |

### 5.2 `big_matrix` Holdout Evaluation Results

* **Ranking Metrics (Relevance: `watch_ratio > 1.0`):**

| Metric        | @5     | @10    | @20    | @50    | @100   | @250   |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Avg Precision | 0.6859 | 0.6534 | 0.6143 | 0.5339 | 0.4491 | 0.3122 |
| Avg Recall    | 0.0669 | 0.1220 | 0.2139 | 0.4128 | 0.6123 | 0.8831 |
| Avg NDCG      | 0.7668 | 0.7508 | 0.7326 | 0.7291 | 0.7781 | 0.9156 |

### 5.3 Discussion of Results

* **Strong Performance on Known Data (`small_matrix`):**

  * The LightGBM ranker demonstrates excellent precision and nDCG on the small_matrix. This indicates it has effectively learned to identify and rank highly engaging items for users and items it was exposed to during training (as this cohort is part of big_matrix). 

  * The low recall is an artifact of the dense evaluation setup, where each user has many "relevant" items, making it hard to capture a large fraction in a short top-K list. The MAE of ~0.35 suggests the watch_ratio predictions are, on average, reasonably close to the actuals.

* **Generalization to Unseen Data (`big_matrix` test):**

  * As expected, precision and nDCG are lower on the big_matrix test set compared to the small_matrix. This reflects the increased difficulty of predicting for chronologically newer interactions, which may involve less familiar items or evolving user preferences.
  
  * Recall is significantly higher on the big_matrix test set. This is because users in the test set have a smaller number of actual positive interactions (compared to the ~3300 items per user in the small_matrix scope), making it easier to recall a larger fraction of these true positives.

* **Overall:**
  The model shows strong learning capabilities. The difference in metrics between the two evaluation sets highlights the importance of evaluating on data that mirrors the deployment scenario (he big_matrix holdout).

---

## 6. Conclusions and Future Work

### 6.1 Conclusions

* Feature engineering—especially for daily item features—was critical for efficiency and accuracy.
* LightGBM shows excellent ranking performance on known cohorts and reasonable performance on unseen data.
* `small_matrix` is a useful validation tool.
* L1 loss was a suitable choice for `watch_ratio` prediction.

### 6.2 Future Work

* **End-to-End Evaluation:**
  Implement full ALS → LightGBM pipeline on test set.

* **Hyperparameter Tuning:**
  Tune ALS and LightGBM using a validation split from training data.

* **Advanced Feature Engineering:**

  * *Embeddings:* For users, items, tags, authors, etc.
  * *Interaction History:* E.g., avg user engagement with similar items.
  * *Social Network:* Integrate `social_network.csv` (e.g., friends' interactions).

* **Alternative Candidate Generators:**
  Try item-based CF or two-tower models.

* **Alternative Ranking Models:**
  Explore deep learning-based ranking models (e.g., neural nets, transformers).
