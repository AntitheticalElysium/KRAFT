# KRAFT
KuaiRec Recommendation Algorithm for Feed Tailoring

**Objective:** Develop a recommender system to suggest short videos to users based on user preferences, interaction histories, and video content using the KuaiRec dataset. The goal is to create a personalized and scalable recommendation engine.

## 1. Introduction

This project implements a multi-stage hybrid recommender system for short videos, leveraging the KuaiRec dataset. The system aims to provide personalized video suggestions by first generating a broad set of candidate videos for each user and then re-ranking these candidates using a more sophisticated model that incorporates rich user, item, and interaction features.

The KuaiRec dataset is unique as its `small_matrix` is described as "fully observed," meaning nearly every user has an interaction record (with a `watch_ratio`) for every item. This makes the task primarily one of **ranking** items based on predicted engagement, rather than predicting missing interactions.

---

## 2. Project Setup

### Installation: 

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh

3. Methodology

The project follows a structured approach, divided into several key stages implemented across Jupyter notebooks.
Data Preprocessing & Preparation (Notebook 1_Data_Preparation.ipynb)

This notebook is responsible for transforming the raw KuaiRec data into a format suitable for modeling and evaluation.

    Load Raw Data: small_matrix.csv (interactions) and item_categories.csv (item metadata) are loaded.

    Create video_metadata.csv: item_id (renamed from video_id) and feat (item category/tag list) are extracted from item_categories.csv.

    Process Interactions:

        video_id is renamed to item_id for consistency.

        A positive_interaction flag (target variable) is derived from the watch_ratio. Based on EDA and dataset description, a watch_ratio >= 1.0 was chosen, implying the user watched the video to completion. This choice aims to capture significant engagement.

        Interactions are sorted chronologically by timestamp.

    Time-Based Split: The sorted interactions are split into training (80%) and testing (20%) sets to simulate a real-world scenario where we predict future interactions.

    Generate Output Files:

        interactions_train.csv: Contains user_id, item_id, watch_ratio, timestamp, and positive_interaction for the training period.

        interactions_test.csv: Contains user_id, item_id pairs from the test period for which predictions will be made. Users not present in the training set are filtered out.

        sample_submission.csv: A template file with user_id, item_id, and a placeholder score.

        test_user_item_map.pkl: A pickled Python dictionary mapping user_id to a set of item_ids they positively interacted with in the test period. This serves as the ground truth for evaluation.

Exploratory Data Analysis (EDA) (Notebook 2_EDA.ipynb)

This notebook analyzes the processed interactions_train.csv and video_metadata.csv.

    Interaction Statistics: Analyzed the number of unique users, items, interactions, and data sparsity.

    Distribution Analysis: Investigated the distribution of interactions per user and per item, revealing typical long-tail distributions. This highlights potential popularity bias.

    Interaction Signal: Examined the distribution of watch_ratio and the balance of the positive_interaction label. This confirmed the chosen threshold for positive interaction resulted in an imbalanced dataset (common in recommendations).

    Temporal Patterns: Plotted interactions per day to observe any trends or seasonality.

    Metadata Analysis: Explored the feat column (item categories), including the number of categories per item and the frequency of individual category IDs.

EDA confirmed the data's characteristics (sparsity in training set, long tails, label imbalance) which informed the modeling strategy.
Model Development (Notebook 3_Model_Training.ipynb)

A multi-stage hybrid recommendation approach was adopted:
Multi-Stage Recommendation Approach

    Stage 1: Candidate Generation: A model generates a larger set of potentially relevant items for each user.

    Stage 2: Ranking: A more complex model re-ranks these candidates using richer features to produce the final, ordered recommendation list.

Stage 1: Candidate Generation (ALS)

    Model Choice: Alternating Least Squares (ALS) from the implicit library was chosen. ALS is effective for implicit feedback data and can efficiently learn user and item latent factor embeddings.

    Data Preparation:

        User and item IDs from interactions_train.csv were mapped to contiguous 0-based integer indices using sklearn.preprocessing.LabelEncoder. These encoders are saved.

        A sparse user-item interaction matrix was created using positive_interaction as the signal (1 for positive, 0 for non-positive based on the threshold). The matrix dimensions are (number of users x number of items).

    Training: The ALS model was trained on this sparse matrix.

        Hyperparameters (example): factors=64, regularization=0.01, iterations=20.

    Output & Saving: The trained ALS model, and the learned user and item embedding vectors, were saved to disk. These embeddings serve as crucial features for the ranking stage.

Stage 2: Ranking (LightGBM)

    Model Choice: LightGBM, a gradient boosting framework, was chosen for its efficiency, ability to handle large datasets, and good performance with categorical features. The task is framed as a binary classification problem: predicting the positive_interaction flag.

    Feature Engineering for Ranker:
    A rich feature set was constructed for each user-item pair in interactions_train.csv:

        User Features (from user_features.csv):

            user_active_degree (categorical)

            is_lowactive_period, is_live_streamer, is_video_author (binary/numerical)

            follow_user_num, fans_user_num, friend_user_num, register_days (numerical)

            onehot_feat0 to onehot_feat17 (categorical)

        Item Features (from video_metadata.csv):

            num_categories: Number of categories associated with the item (derived from feat).

        Collaborative Features (from ALS):

            User embedding vector (64 dimensions).

            Item embedding vector (64 dimensions).

        Interaction Count Features:

            user_interaction_count: Total interactions for the user in the training set.

            item_interaction_count: Total interactions for the item in the training set.

    Data Preparation for LightGBM:

        All features were merged onto the interactions_train_df.

        Missing values were imputed (0 for numerical, -1 for categorical features before astype('category')). This strategy was chosen for simplicity and to handle NaNs introduced during left merges if users/items had no external features or embeddings (e.g., items only in interactions but not in metadata, or users not in user_features.csv).

        Categorical features (like user_active_degree and onehot_feat*) were converted to category dtype, which LightGBM handles natively.

    Training:

        The feature-engineered dataset was split into training (80%) and validation (20%) sets for LightGBM using train_test_split with stratification on the target.

        The LightGBM classifier was trained using parameters like objective='binary', metric='auc', n_estimators=1000 (with early stopping on validation AUC).

    Output & Saving: The trained LightGBM model and the list of feature columns used for training were saved.

Evaluation & Submission (Notebook 4_Evaluation_Submission.ipynb)

    Load Models & Data: Trained ALS (for embeddings), LightGBM ranker, encoders, feature list, interactions_test.csv, and test_user_item_map.pkl (ground truth) were loaded.

    Prepare Test Features: The exact same feature engineering and imputation steps applied to the training data were replicated for the interactions_test.csv pairs. This ensures consistency.

    Generate Predictions: The LightGBM ranker predicted the probability of positive interaction (score) for each user-item pair in the test set.

    Evaluation Metrics:
    The performance of the ranker was evaluated using standard ranking metrics @K (K=10, 20, 50). These metrics are meaningful for recommendation tasks as they assess the quality of the top-N ranked list.

        Precision@K: Proportion of recommended items in the top-K that are relevant.

        Recall@K: Proportion of all relevant items (ground truth for a user) that are found in the top-K recommendations.

        NDCG@K (Normalized Discounted Cumulative Gain): Measures ranking quality by considering the position of relevant items, giving higher scores if relevant items are ranked higher.
        The utils.py script contains functions for calculating these metrics.

    Create Submission File: The submission.csv file was generated with user_id, item_id, and the predicted score.

4. Experiments & Results
Defining Positive Interaction

The watch_ratio was used as the primary engagement signal. A watch_ratio >= 1.0 was chosen to define a positive_interaction.

    Justification: This threshold implies the user viewed at least the entire video. EDA on interactions_train.csv showed this captured ~32.79% of interactions as positive, providing a reasonable (though imbalanced) target for the binary classification ranker.

    (Optional: Add any experiments if you tried other thresholds and their impact)

Model Performance

The multi-stage hybrid model achieved the following results on the test set (evaluated on 1411 users with ground truth):
Metric	@10	@20	@50
Avg Precision	0.7924	0.7529	0.6670
Avg Recall	0.0460	0.0858	0.1819
Avg NDCG	0.8110	0.7772	0.7036

(Note: Ensure these results match your latest run on the full training data.)
Handling Memory Constraints

During development, memory issues were encountered, particularly during feature engineering for the LightGBM ranker (due to the large size of ranker_train_df with ~3.7M rows and many features) and during LightGBM training itself. These were addressed by:

    Feature Engineering Optimization:

        Downcasting data types (e.g., float64 to float32 for embeddings, int64 to smaller integer types).

        Explicitly deleting large intermediate DataFrames and using gc.collect().

        Calculating interaction counts more efficiently using map(groupby().size()) instead of transform('size').

    LightGBM Parameter Tuning for Memory:

        Setting max_bin (e.g., to 128) in LightGBM parameters helped reduce memory during training.

        (If you had to train on a sample, mention it here: e.g., "Initially, training on the full dataset still exceeded memory. The final ranker model was trained on a 50% random sample of the prepared ranker training data to fit within resource constraints. This was documented and considered during result interpretation.")

        It was found that with careful feature engineering and the max_bin adjustment, training on the full dataset became feasible.

5. Discussion & Conclusions
Functionality & Relevance

The recommender system successfully generates personalized video suggestions.

    High-Quality Suggestions: The high Precision@K (e.g., ~79% @10) and NDCG@K (e.g., ~0.81 @10) indicate that the items ranked highest by the model are very likely to be relevant and are ordered effectively. Users are likely to find engaging content quickly.

    Relevance: The model leverages collaborative patterns (through ALS embeddings) and user/item-specific features, allowing it to make predictions tailored to individual user profiles and item characteristics.

Accuracy & Metrics

The chosen metrics (Precision@K, Recall@K, NDCG@K) are standard and meaningful for evaluating ranking performance in recommender systems.

    Precision & NDCG: The model demonstrates strong performance in these areas, signifying accurate top-N recommendations and good ranking quality.

    Recall: The recall is relatively low (~18% @50). This suggests that while the top recommendations are good, the system doesn't capture the full breadth of items a user might positively interact with in the top 50. Given the "fully observed" nature of the original small_matrix, this implies users have a large set of positively viewed items in the test period, and the challenge is to prioritize the best among them effectively and also ensure a diverse set of these liked items appear in the top recommendations.

Key Learnings & Challenges

    Memory Management: Handling large datasets effectively is crucial. Techniques like downcasting, deleting unused objects, and optimizing calculations were vital.

    Feature Engineering: The inclusion of ALS embeddings as features in the LightGBM ranker significantly contributes to its performance by injecting collaborative signals.

    Multi-Stage Approach: The separation of candidate generation and ranking is an effective strategy for balancing computational efficiency and recommendation quality.

    "Fully Observed" Data: Understanding this characteristic of KuaiRec shifts the problem emphasis more towards ranking items already seen (with varying watch ratios) rather than discovering unseen items.

Potential Future Improvements

    Enhance Recall:

        Diversify Candidate Generation: Incorporate more diverse candidate sources (content-based, popularity-based from different niches, new items) alongside ALS.

        Increase Candidate Pool Size: Provide the ranker with a larger set of initial candidates.

        Re-ranking for Diversity: Implement MMR or category-based diversification after the LightGBM scoring to ensure a wider variety of liked items appear in the top-N.

    Advanced Feature Engineering:

        Utilize text features from kuairec_caption_category.csv (e.g., TF-IDF, embeddings).

        Incorporate more detailed item features from item_daily_features.csv (e.g., author popularity, item age, carefully aggregated daily stats).

        Explore more complex interaction features (e.g., user's affinity for item categories/authors over time).

    Model Tuning & Exploration:

        Systematic hyperparameter tuning for both ALS and LightGBM.

        Experiment with different LightGBM objectives (e.g., lambdarank if formulating as a learning-to-rank problem).

        Explore neural network-based rankers (DeepFM, Wide & Deep) which can automatically learn feature interactions.

    Address Popularity Bias: More explicitly penalize overly popular items during re-ranking or explore techniques to de-bias the training data or model.

    Online Evaluation Simulation: If more fine-grained temporal data were available per user, one could simulate an online evaluation setting to better assess session-based performance.
