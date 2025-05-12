# KRAFT : KuaiRec Recommendation Algorithm for Feed Tailoring

**Objective:** Develop a recommender system to suggest short videos to users based on user preferences, interaction histories, and video content using the KuaiRec dataset. The goal is to create a personalized and scalable recommendation engine.


## 1. Introduction

This project implements a multi-stage hybrid recommender system for short videos, leveraging the KuaiRec dataset. The system aims to provide personalized video suggestions by first generating a broad set of candidate videos for each user and then re-ranking these candidates using a more sophisticated model that incorporates rich user, item, and interaction features.

The KuaiRec dataset is unique as its `small_matrix` is described as "fully observed," meaning nearly every user has an interaction record (with a `watch_ratio`) for every item. This makes the task primarily one of **ranking** items based on predicted engagement, rather than predicting missing interactions.


## 2. Project Setup

### Installation: 

```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

bash scripts/download_data.sh
```

## 3. Methodology

### Data Preprocessing & Preparation (Notebook `1_Data_Preparation.ipynb`)

This notebook is responsible for transforming the raw KuaiRec data into a format suitable for modeling and evaluation.

*   **Load Raw Data:** `small_matrix.csv` (interactions) and `item_categories.csv` (item metadata) are loaded.
*   **Create `video_metadata.csv`:** `item_id` (renamed from `video_id`) and `feat` (item category/tag list) are extracted from `item_categories.csv`.
*   **Process Interactions:**
    *   `video_id` is renamed to `item_id` for consistency.
    *   A `positive_interaction` flag (target variable) is derived from the `watch_ratio`. Based on EDA and dataset description, a `watch_ratio >= 1.0` was chosen, implying the user watched the video to completion. This choice aims to capture significant engagement.
    *   Interactions are sorted chronologically by `timestamp`.
*   **Time-Based Split:** The sorted interactions are split into training (80%) and testing (20%) sets to simulate a real-world scenario where we predict future interactions.
*   **Generate Output Files:**
    *   `interactions_train.csv`: Contains `user_id`, `item_id`, `watch_ratio`, `timestamp`, and `positive_interaction` for the training period.
    *   `interactions_test.csv`: Contains `user_id`, `item_id` pairs from the test period for which predictions will be made. Users not present in the training set are filtered out.
    *   `sample_submission.csv`: A template file with `user_id`, `item_id`, and a placeholder `score`.
    *   `test_user_item_map.pkl`: A pickled Python dictionary mapping `user_id` to a set of `item_id`s they positively interacted with in the test period. This serves as the ground truth for evaluation.

### Exploratory Data Analysis (EDA) (Notebook `2_EDA.ipynb`)

This notebook analyzes the *processed* `interactions_train.csv` and `video_metadata.csv`.

*   **Interaction Statistics:** Analyzed the number of unique users, items, interactions, and data sparsity.
*   **Distribution Analysis:** Investigated the distribution of interactions per user and per item, revealing typical long-tail distributions. This highlights potential popularity bias.
*   **Interaction Signal:** Examined the distribution of `watch_ratio` and the balance of the `positive_interaction` label. This confirmed the chosen threshold for positive interaction resulted in an imbalanced dataset (common in recommendations).
*   **Temporal Patterns:** Plotted interactions per day to observe any trends or seasonality.
*   **Metadata Analysis:** Explored the `feat` column (item categories), including the number of categories per item and the frequency of individual category IDs.

EDA confirmed the data's characteristics (sparsity in training set, long tails, label imbalance) which informed the modeling strategy.

### Model Development (Notebook `3_Model_Training.ipynb`)

A **multi-stage hybrid recommendation approach** was adopted:

#### Multi-Stage Recommendation Approach
1.  **Stage 1: Candidate Generation:** A model generates a larger set of potentially relevant items for each user.
2.  **Stage 2: Ranking:** A more complex model re-ranks these candidates using richer features to produce the final, ordered recommendation list.

#### Stage 1: Candidate Generation (ALS)
1.  **Model Choice:** Alternating Least Squares (ALS) from the `implicit` library was chosen. ALS is effective for implicit feedback data and can efficiently learn user and item latent factor embeddings.
2.  **Data Preparation:**
    *   User and item IDs from `interactions_train.csv` were mapped to contiguous 0-based integer indices using `sklearn.preprocessing.LabelEncoder`. These encoders are saved.
    *   A sparse user-item interaction matrix was created using `positive_interaction` as the signal (1 for positive, 0 for non-positive based on the threshold). The matrix dimensions are (number of users x number of items).
3.  **Training:** The ALS model was trained on this sparse matrix.
4.  **Output & Saving:** The trained ALS model, and the learned user and item embedding vectors, were saved to disk. These embeddings serve as crucial features for the ranking stage.

#### Stage 2: Ranking (LightGBM)
1.  **Model Choice:** LightGBM, a gradient boosting framework, was chosen for its efficiency, ability to handle large datasets, and good performance with categorical features. The task is framed as a binary classification problem: predicting the `positive_interaction` flag.
2.  **Feature Engineering for Ranker:**
    A rich feature set was constructed for each user-item pair in `interactions_train.csv`:
    *   **User Features (from `user_features.csv`):**
        *   `user_active_degree` (categorical)
        *   `is_lowactive_period`, `is_live_streamer`, `is_video_author` (binary/numerical)
        *   `follow_user_num`, `fans_user_num`, `friend_user_num`, `register_days` (numerical)
        *   `onehot_feat0` to `onehot_feat17` (categorical)
    *   **Item Features (from `video_metadata.csv`):**
        *   `num_categories`: Number of categories associated with the item (derived from `feat`).
    *   **Collaborative Features (from ALS):**
        *   User embedding vector (64 dimensions).
        *   Item embedding vector (64 dimensions).
    *   **Interaction Count Features:**
        *   `user_interaction_count`: Total interactions for the user in the training set.
        *   `item_interaction_count`: Total interactions for the item in the training set.
3.  **Data Preparation for LightGBM:**
    *   All features were merged onto the `interactions_train_df`.
    *   Missing values were imputed (0 for numerical, -1 for categorical features before `astype('category')`). This strategy was chosen for simplicity and to handle NaNs introduced during left merges if users/items had no external features or embeddings (e.g., items only in interactions but not in metadata, or users not in `user_features.csv`).
    *   Categorical features (like `user_active_degree` and `onehot_feat*`) were converted to `category` dtype, which LightGBM handles natively.
4.  **Training:**
    *   The feature-engineered dataset was split into training (80%) and validation (20%) sets for LightGBM using `train_test_split` with stratification on the target.
    *   The LightGBM classifier was trained using parameters like `objective='binary'`, `metric='auc'`, `n_estimators=1000` (with early stopping on validation AUC).
5.  **Output & Saving:** The trained LightGBM model and the list of feature columns used for training were saved.

### Evaluation & Submission (Notebook `4_Evaluation_Submission.ipynb`)

1.  **Load Models & Data:** Trained ALS (for embeddings), LightGBM ranker, encoders, feature list, `interactions_test.csv`, and `test_user_item_map.pkl` (ground truth) were loaded.
2.  **Prepare Test Features:** The *exact same* feature engineering and imputation steps applied to the training data were replicated for the `interactions_test.csv` pairs. This ensures consistency.
3.  **Generate Predictions:** The LightGBM ranker predicted the probability of positive interaction (`score`) for each user-item pair in the test set.
4.  **Evaluation Metrics:**
    The performance of the ranker was evaluated using standard ranking metrics @K (K=10, 20, 50). These metrics are meaningful for recommendation tasks as they assess the quality of the top-N ranked list.
    *   **Precision@K:** Proportion of recommended items in the top-K that are relevant.
    *   **Recall@K:** Proportion of all relevant items (ground truth for a user) that are found in the top-K recommendations.
    *   **NDCG@K (Normalized Discounted Cumulative Gain):** Measures ranking quality by considering the position of relevant items, giving higher scores if relevant items are ranked higher.
    The `utils.py` script contains functions for calculating these metrics.
5.  **Create Submission File:** The `submission.csv` file was generated with `user_id`, `item_id`, and the predicted `score`.


## 4. Experiments & Results

### Defining Positive Interaction
The `watch_ratio` was used as the primary engagement signal. A `watch_ratio >= 1.0` was chosen to define a `positive_interaction`.
*   **Justification:** This threshold implies the user viewed at least the entire video. EDA on `interactions_train.csv` showed this captured ~32.79% of interactions as positive, providing a reasonable (though imbalanced) target for the binary classification ranker.

### Model Performance
The multi-stage hybrid model (ALS for candidate generation, LightGBM for ranking) was evaluated on a test set comprising 20% of the interaction data, split chronologically. The evaluation was performed on 1411 users for whom ground truth positive interactions were available in the test period.

The following table summarizes the average ranking metrics achieved:

| Metric         | @10    | @20    | @50    | @100   | @500   | @1000  |
|----------------|--------|--------|--------|--------|--------|--------|
| Avg Precision  | 0.7924 | 0.7529 | 0.6670 | 0.5468 | 0.3103 | 0.2043 |
| Avg Recall     | 0.0460 | 0.0858 | 0.1819 | 0.2821 | 0.7620 | 1.0000 |
| Avg NDCG       | 0.8110 | 0.7772 | 0.7036 | 0.6042 | 0.7014 | 0.8476 |

**Observations from Results:**

*   **High Top-K Precision:** The model exhibits strong precision at smaller values of K, with approximately 79% of the top 10 recommendations being relevant. This indicates that users are very likely to find engaging content at the beginning of their feed.
*   **Excellent Ranking Quality (NDCG):** The NDCG scores are notably high, especially NDCG@10 at 0.8110. This signifies that not only are relevant items present in the top recommendations, but they are also ranked appropriately (more relevant items appear higher). Interestingly, NDCG@500 and NDCG@1000 also show strong ranking quality across a larger set of items.
*   **Recall Improvement with K:** Recall starts low but increases significantly as K grows.
    *   Recall@50 reaches ~18%, indicating that about one-fifth of all items a user liked in the test set are found within the top 50 recommendations.
    *   Recall@500 impressively reaches ~76%, meaning the model can retrieve over three-quarters of the user's liked items if a larger recommendation slate is considered.
    *   Recall@1000 achieves 1.0000, implying that within the top 1000 ranked items (from the `interactions_test.csv` pairs evaluated for each user), all ground truth positive items for those users were captured. This is expected given the "fully observed" nature of the small matrix, as all items were available to be scored. The challenge lies in ranking them correctly.
*   **Precision-Recall Trade-off:** As expected, precision decreases as K increases, while recall increases. The significant jump in recall at K=500 suggests that many relevant items are scored reasonably well by the ranker but don't make it into the very top positions.

### Handling Memory Constraints
During development, memory issues were encountered, particularly during feature engineering for the LightGBM ranker (due to the large size of `ranker_train_df` with ~3.7M rows and many features) and during LightGBM training itself. These were addressed by:
1.  **Feature Engineering Optimization:**
    *   Downcasting data types (e.g., `float64` to `float32` for embeddings, `int64` to smaller integer types).
    *   Explicitly deleting large intermediate DataFrames and using `gc.collect()`.
    *   Calculating interaction counts more efficiently using `map(groupby().size())` instead of `transform('size')`.
2.  **LightGBM Parameter Tuning for Memory:**
    *   Setting `max_bin = 128` in LightGBM parameters helped reduce memory during training.
    *   With these optimizations, training on the full dataset became feasible.
