# FIFA World Cup 2022 Sentiment Analysis Assignment

## (1) Problem Statement
The task involves analyzing public sentiment regarding the FIFA World Cup 2022 using Twitter data. Social media provides a vast amount of unstructured text that reflects public opinion, but categorizing this into meaningful sentiments (Positive, Neutral, Negative) manually is impossible at scale.

## (2) Objective
To develop and compare machine learning classifiers (Na&iuml;ve Bayes, Logistic Regression, and SVM) that can automatically categorize the sentiment of FIFA World Cup 2022 tweets based on their text content.

## (3) Dataset
- **Source:** `/content/fifa_world_cup_2022_tweets.csv`
- **Features:** `Tweet` (textual content), `Sentiment` (label).
- **Size:** Original dataset contains 100 rows; analysis performed on a subset of 100 samples with an 80/20 train-test split.

## (4) Methodology
- **Data Preprocessing:** Converted text to lowercase, removed URLs, user mentions (@), hashtags (#), and non-alphabetic characters. Numeric labels were mapped (Positive: 1, Neutral: 0, Negative: -1).
- **EDA:** Checked class distributions; observed that the dataset is imbalanced with more positive than negative samples.
- **Model Building:** Used TF-IDF Vectorization (unigrams and bigrams). Implemented three models: Multinomial Na&iuml;ve Bayes, Logistic Regression (balanced), and Linear SVM (balanced).
- **Evaluation:** Evaluated using Accuracy, Weighted Precision, Weighted Recall, and Confusion Matrices.

## (5) Results
Based on the latest execution:
- **Na&iuml;ve Bayes:** ~50% Accuracy. Best at recalling Positive tweets but struggled with minority classes.
- **Logistic Regression:** ~40% Accuracy. Provided a balanced attempt at classification across labels.
- **SVM:** ~35% Accuracy. Performance was limited by the small sample size (100 rows).

## (6) How to Run
1. Ensure the dataset is placed at `/content/fifa_world_cup_2022_tweets.csv`.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the notebook cells sequentially to train models and generate visualizations.

## (7) Conclusion
The Na&iuml;ve Bayes model performed the best in terms of overall accuracy for this specific subset. However, the model performance is significantly affected by the small training size (80 samples). For better generalization, a larger dataset and more advanced NLP techniques (like Word Embeddings or Transformers) could be utilized.

## (8) Student's details
- **Name:** Ahad Khan
- **Roll No:** 24
- **UIN:** 242A006
- **YEAR:** TE-AIDS
