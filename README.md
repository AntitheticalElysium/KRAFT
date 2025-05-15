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

3. Methodology

The recommender system employs a two-stage architecture, a common pattern for balancing scalability and prediction accuracy in large-scale systems:
3.1. Data Preparation and Feature Engineering

This crucial phase (detailed in notebooks/1_Data_Preparation.ipynb) involves:

    Loading Raw Data: Ingesting big_matrix.csv, small_matrix.csv, user_features.csv, item_categories.csv, and item_daily_features.csv.

    Initial Cleaning: Handling missing values, correcting data types, and parsing complex string fields (e.g., item tags).

    Feature Creation:

        Temporal Features: Extracting interaction_hour and interaction_day_of_week from interaction timestamps.

        User Features: Utilizing selected features like user_active_degree, fans_user_num, register_days, is_lowactive_period, is_live_streamer, is_video_author, and pre-encoded onehot_feat*.

        Static Item Features: Deriving num_item_tags from item_categories.csv.

        Dynamic Item Features: Calculating video_age_days (from upload_dt), and key daily engagement ratios (daily_play_per_show_ratio, daily_like_per_play_ratio, daily_completion_rate) from item_daily_features.csv. Selected raw daily stats like daily_play_progress, video_duration_daily, author_id, video_type, and daily video_tag_id are also incorporated.

        Duplicate Handling: A critical step involved identifying and removing duplicate entries in item_daily_features based on (video_id, date) to prevent data explosion during merges.

    Data Splitting & Saving:

        The big_matrix is chronologically split into training (80%) and testing (20%) sets for the LightGBM ranker.

        The feature-engineered small_matrix is saved separately for dense evaluation.

        Processed data is saved in Parquet format for efficiency.

3.2. Stage 1: Candidate Generation (ALS)

    Model: Alternating Least Squares (ALS) from the implicit library is used. ALS is a matrix factorization technique well-suited for implicit feedback data.

    Training Data: Trained on the user-item interaction matrix derived from the full big_matrix, using watch_ratio (clipped > 0) as the confidence score.

    Output: For a given user, ALS generates a list of candidate item IDs that the user might be interested in. This stage narrows down the vast item catalog to a manageable set for the more computationally intensive ranking stage.

    Implementation: See notebooks/2_Model_Training.ipynb.

3.3. Stage 2: Ranking (LightGBM)

    Model: LightGBM, a gradient boosting framework, is used as the ranker. It's chosen for its efficiency, scalability, and ability to handle large, sparse datasets with a mix of categorical and numerical features.

    Task: It's trained as a regression model to predict the watch_ratio for a given (user, item) pair, using the rich feature set created in the data preparation phase.

    Training Data: Trained on the 80% training split of the feature-engineered big_matrix.

    Loss Function: L1 loss (Mean Absolute Error) is used, which is generally more robust to outliers in the watch_ratio target variable.

    Output: For each (user, candidate_item) pair, LightGBM outputs a predicted watch_ratio. These scores are then used to re-rank the candidates from Stage 1.

    Implementation: See notebooks/2_Model_Training.ipynb.

4. Experiments and Evaluation

Two primary evaluation scenarios were conducted (detailed in notebooks/3_Model_Evaluation.ipynb):
4.1. Evaluation on small_matrix (Dense Subset)

    Purpose: To assess the LightGBM ranker's performance on a familiar, dense dataset where users and items were part of the big_matrix training data. This provides a clean measure of pointwise prediction accuracy and ranking quality on a known cohort.

    Methodology:

        The fully feature-engineered small_matrix data was used.

        For each user in small_matrix, the LightGBM model predicted watch_ratio for all items they interacted with in this matrix.

        Items were ranked based on these predicted scores.

    Metrics:

        Pointwise: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) for watch_ratio prediction.

        Ranking: Precision@k, Recall@k, nDCG@k (for k = 5, 10, 20, 50, 100, 250), using watch_ratio > 1.0 as the relevance threshold.

4.2. Evaluation on big_matrix Holdout Set

    Purpose: To assess the LightGBM ranker's performance on unseen future interactions from the main dataset distribution.

    Methodology (Current Implementation):

        The 20% chronological test split from the feature-engineered big_matrix was used.

        For each user, the LightGBM model predicted watch_ratio for all items they actually interacted with in this test set.

        Items were ranked based on these predicted scores.

        Note: A true end-to-end evaluation (ALS candidate generation followed by LightGBM re-ranking on those candidates) is a more complete test for this scenario but was simplified for this iteration to focus on the ranker's performance on test set items.

    Metrics:

        Ranking: Precision@k, Recall@k, nDCG@k.

5. Results
   
5.1. small_matrix Evaluation Results:

    Pointwise Metrics:

        Overall RMSE: 1.3214

        Overall MAE: 0.3473

    Ranking Metrics (Relevance: watch_ratio > 1.0):
| Metric        | @5     | @10    | @20    | @50    | @100   | @250   |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Avg Precision | 0.9137 | 0.9117 | 0.9019 | 0.8571 | 0.8234 | 0.7725 |
| Avg Recall    | 0.0048 | 0.0095 | 0.0188 | 0.0443 | 0.0848 | 0.1973 |
| Avg NDCG      | 0.9146 | 0.9128 | 0.9057 | 0.8706 | 0.8396 | 0.7904 |


5.2. big_matrix Holdout Evaluation Results (LGBM ranking on test items):

    Ranking Metrics (Relevance: watch_ratio > 1.0):

| Metric        | @5     | @10    | @20    | @50    | @100   | @250   |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Avg Precision | 0.6859 | 0.6534 | 0.6143 | 0.5339 | 0.4491 | 0.3122 |
| Avg Recall    | 0.0669 | 0.1220 | 0.2139 | 0.4128 | 0.6123 | 0.8831 |
| Avg NDCG      | 0.7668 | 0.7508 | 0.7326 | 0.7291 | 0.7781 | 0.9156 |

5.3. Discussion of Results:

    Strong Performance on Known Data (small_matrix): The LightGBM ranker demonstrates excellent precision and nDCG on the small_matrix. This indicates it has effectively learned to identify and rank highly engaging items for users and items it was exposed to during training (as this cohort is part of big_matrix). The low recall is an artifact of the dense evaluation setup, where each user has many "relevant" items, making it hard to capture a large fraction in a short top-K list. The MAE of ~0.35 suggests the watch_ratio predictions are, on average, reasonably close to the actuals.

    Performance on Unseen Interactions (big_matrix test):

        As expected, precision and nDCG are lower on the big_matrix test set compared to the small_matrix. This reflects the increased difficulty of predicting for chronologically newer interactions, which may involve less familiar items or evolving user preferences. P@5 of ~0.69 is still a respectable result.

        Recall is significantly higher on the big_matrix test set. This is because users in the test set have a smaller number of actual positive interactions (compared to the ~3300 items per user in the small_matrix scope), making it easier to recall a larger fraction of these true positives.

    Overall: The model shows strong learning capabilities. The difference in metrics between the two evaluation sets highlights the importance of evaluating on data that mirrors the deployment scenario (i.e., the big_matrix holdout).

6. Conclusions and Future Work
6.1. Conclusions:

    The implemented feature engineering, particularly the handling of daily item features (ratios, duplicate removal), was crucial for managing memory and creating an effective LightGBM ranker.

    The LightGBM model demonstrates strong ranking capabilities, especially on data it is familiar with. Its performance on the big_matrix holdout is reasonable and provides a solid baseline.

    The small_matrix serves as an excellent tool for detailed validation of the ranker on a dense, known cohort, confirming its ability to learn engagement patterns.

    The choice of L1 loss for the watch_ratio regression appears justified given the nature of the target.

6.2. Future Work:

    Full End-to-End Evaluation: Implement the complete two-stage evaluation on the big_matrix test set: ALS candidate generation followed by LightGBM re-ranking of those candidates. This will provide the most realistic measure of overall system performance.

    Hyperparameter Tuning: Systematically tune hyperparameters for both ALS (factors, regularization, iterations) and LightGBM (num_leaves, learning_rate, feature_fraction, etc.) using a proper validation set derived from the big_matrix training data.

    Advanced Feature Engineering:

        Embeddings: Incorporate pre-trained or jointly learned embeddings for users, items, authors, tags, etc., as features for LightGBM.

        Interaction Features: Create features representing user-item interaction history (e.g., user's average watch ratio on previously seen items by the same author, or in the same category).

        Social Network Features: Explore ways to integrate information from social_network.csv (e.g., features based on friends' interactions).

    Explore Different Candidate Generators: Experiment with other candidate generation techniques beyond ALS (e.g., item-based collaborative filtering, two-tower neural models).

    Alternative Ranking Models: Evaluate more complex ranking models, such as deep learning-based rankers (e.g., DeepFM, Wide & Deep), if computational resources allow.

    Cold-Start Strategies: Develop and evaluate specific strategies for handling new users and new items more effectively.

    Diversity and Serendipity: Incorporate objectives or post-processing steps to improve the diversity and serendipity of recommendations, beyond pure accuracy/engagement.
    *   Setting `max_bin = 128` in LightGBM parameters helped reduce memory during training.
    *   With these optimizations, training on the full dataset became feasible.
