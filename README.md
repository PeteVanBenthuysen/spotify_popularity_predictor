# **Spotify Song Popularity Prediction**

## **Goal**
The goal of this project is to **predict whether a song is popular** on Spotify, with popularity defined as a score greater than or equal to **59**. By understanding the factors that influence a song’s popularity, we aim to identify what variables cause a song to go mainstream, making this a valuable tool for artists, producers, and marketers who wish to optimize song appeal.

## **Why This Goal?**
Getting a song played on mainstream radio or reaching a large audience on streaming platforms is **not an easy task**. There are numerous factors—such as tempo, danceability, energy, and other audio features—that contribute to a song's success. By analyzing these features, we aim to build a model that can predict whether a song is likely to become popular, based on its characteristics.

## **Project Requirements**
This project requires the following Python libraries for data analysis, modeling, and visualization:

```plaintext
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.3.2
matplotlib==3.8.4
seaborn==0.13.2
jupyter==1.0.0
imbalanced-learn==0.11.0   
```

## Dataset

The dataset used in this project is the **Ultimate Spotify Tracks DB** from Kaggle.

To get started, follow these steps:

1. Go to the [Kaggle dataset page](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db).
2. Download the dataset and unzip it.
3. Place the extracted files in the data/ directory of this project. If the data/ folder does not exist, create it.

## **Loading, Cleaning, and Preprocessing the Dataset**

We began by loading and preprocessing the raw Spotify dataset to ensure that it was clean, consistent, and ready for modeling.

---

### Data Loading
- Loaded the `SpotifyFeatures.csv` dataset from the `data/` directory.
- Disabled date parsing to avoid misinterpreting string columns like `time_signature`.

---

### Cleaning and Deduplication

#### Duplicate Tracks
- Identified and analyzed tracks with duplicate `track_id` values.
- **Found 35,124 duplicated track_ids** with conflicting genres.
- This affected **a total of 91,075 rows**.
- Removed duplicates by keeping only the **first occurrence** of each `track_id`.
- **Removed 55,951 rows** during deduplication.

#### Time Signature Cleanup
- Removed invalid values (`'0/4'`) from the `time_signature` column.
- Extracted the numeric part and converted it to integer format.

#### Duration Fix
- Converted song duration from milliseconds to seconds, creating a new column: `duration_sec` and dropped `duration_ms`.

#### Genre Cleanup
- Replaced invalid genre names (e.g., fixing encoding issues with "Children’s Music").
- Dropped irrelevant genres like **Comedy** and **Children's Music**.
- Merged `"Hip-Hop"` and `"Rap"` into a combined `"Hip-Hop_Rap"` category.

#### Column Removal
- Dropped unnecessary columns such as:
  - `track_name`
  - `track_id`
  - `artist_name`
  - `duration_ms`

---

### Target Variable Creation

We defined a binary target variable `is_popular`:
- Songs in the **top 10% of popularity** (≥ 59.0) were labeled as **1 (popular)**.
- All other songs were labeled as **0 (not popular)**.
- Dropped the original `popularity` column afterward.

---

### Final Checks

- Removed any remaining duplicate rows.
- Verified column data types and structure.
- Confirmed no missing values or nulls in the cleaned dataset.

---

### Final Cleaned Dataset:
- **Shape**: `(159,981 rows × 15 columns)`
- Fully prepared for EDA, feature engineering, and model training.


## **Exploratory Data Analysis (EDA):**
To better understand the dataset and prepare for modeling, we conducted extensive exploratory data analysis:

### 1. Genre Distribution
- Visualized the number of songs in each genre using a horizontal bar chart.
- Merged similar categories (e.g., “Hip-Hop” and “Rap” into “Hip-Hop_Rap”).
- Dropped irrelevant genres like `Comedy` and `Children's Music`.

### 2. Duration Outlier Detection and Removal
- Plotted a histogram of song durations (in seconds) before outlier removal.
- Used the **IQR method** to identify and remove extreme duration values.
- Replotted the cleaned distribution and reported:
  - Number of outliers detected and removed
  - IQR range used for filtering
  - New min/max duration range
  - Final dataset size after cleaning

### 3. Tempo Distribution
- Plotted a histogram of **tempo (BPM)** making sure outliers were not present.

### 4. Energy vs. Popularity (Violin Plot)
- Created a violin plot comparing **energy distribution** between popular and non-popular songs (`is_popular` = 1 vs 0).
- Calculated and printed the IQR for `energy` within each popularity group.

### 5. Feature Interactions: Scatter Plots
- **Danceability vs. Loudness**:
  - Created a scatter plot with correlation coefficient.
  - Included summary statistics for both variables.
- **Loudness vs. Energy**:
  - Repeated scatter plot and correlation analysis.

### 6. Correlation Matrix
- Generated a heatmap to show pairwise correlations among key numerical features:
  - `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`
- Used this matrix to identify multicollinearity and potentially redundant features.

### 7. Summary Checks
- Printed final dataset shape and column names after EDA.
- Reviewed value distributions and confirmed readiness for feature engineering.

---

### Note on Outlier Removal

Outlier removal was **only applied to the `duration_sec` feature**.

This decision was intentional because:
- Many audio features (e.g., energy, danceability, valence, loudness) naturally vary across music genres and styles.
- What may appear as an outlier in one genre (e.g., extremely low energy in Ambient or Classical music) could be typical in that context.
- Removing those values could have biased the model against certain genres or artistic styles.

In contrast, song duration exhibited extreme values (very short or overly long tracks) that were likely to be metadata errors, movies, audiobooks or non-standard releases. Removing those improved data consistency without harming genre diversity.

## **Feature Engineering:**

We engineered new features and transformed existing ones to make the data more suitable for modeling:

### 1. Mode Conversion
- Converted the `mode` column (Major/Minor) to a binary variable:
  - `Major = 1`, `Minor = 0`.

### 2. One-Hot Encoding
- Applied one-hot encoding to the following categorical features:
  - `genre`
  - `key`
  - `time_signature` (prefixed with `ts_`)
  - `tempo_category` (created via binning, see below)

### 3. Vocalness Feature
- Created a new feature called `vocalness`, calculated as:
  - `vocalness = 1 - instrumentalness`
- This captures how vocal-heavy a track is, as the inverse of instrumental content.

### 4. Tempo Binning
- Binned `tempo` into three categories using domain knowledge:
  - `slow`: 0–90 BPM
  - `medium`: 90–120 BPM
  - `fast`: 120+ BPM
- Then one-hot encoded the resulting categories into separate columns.

### 5. Correlation Analysis
- Computed a correlation matrix including the `is_popular` target and core numerical features.
- Visualized relationships with a heatmap to understand which variables are most predictive.

### 6. Redundant Column Removal
- Dropped columns that were no longer needed after transformation:
  - `instrumentalness` (replaced by `vocalness`)
  - `tempo` (replaced by tempo categories)

### 7. Final Dataset Checkpoint
- Verified no missing values or duplicate rows.
- Printed out the dataset shape and column list after all transformations.

- This step prepared the data for splitting and model input by ensuring all features were numeric, relevant, and interpretable.

## **Train/Validation/Test Split**

To properly evaluate model performance and avoid overfitting, the dataset was split into three parts:

### 1. 80/20 Initial Split
- **80%** of the data was used for training and validation (`train_val`).
- **20%** was set aside as a holdout **test set**, never seen during model training or tuning.

### 2. Secondary Split (Train vs. Validation)
- From the `train_val` set, a second split was made:
  - **75% for training**, **25% for validation**.
  - This results in a final 60/20/20 split between **train**, **validation**, and **test** sets.

### 3. Stratified Sampling
- All splits used **stratification on the `is_popular` column** to ensure balanced class proportions across train, validation, and test sets.

### 4. Feature Alignment
- After splitting, feature columns were aligned using `reindex()` to ensure all splits shared the same structure.
- The target column (`is_popular`) was separated from the input features.

### 5. Output Checks
- Printed dataset shapes and class distributions to verify balanced splits.
- Ensured that all columns matched expected features post-split.

## **Feature Scaling & Class Balancing**

To ensure fair model performance and handle class imbalance, we performed both **feature scaling** and **resampling**:

### 1. Standardization (Scaling)
- Applied `StandardScaler` from `scikit-learn` to standardize the input features:
  - Fit only on the **training set** (`X_train`) to avoid data leakage.
  - Transformed **training**, **validation**, and **test** sets using the same scaler.
- This step is essential for models like **Logistic Regression**, which are sensitive to feature magnitudes.

### 2. Class Imbalance Handling with SMOTETomek

We observed that only ~10% of the songs were labeled as "popular", leading to **class imbalance**. To address this, we used:

**SMOTETomek**, which combines:
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Adds synthetic samples to the minority class (popular songs) to balance the class distribution.
- **Tomek Links**: Removes ambiguous samples from the majority class that are close to the decision boundary.

This combined approach:
- **Increases minority representation** (helps recall).
- **Cleans borderline/noisy majority class samples** (improves precision).
- Helps models **learn a more generalizable decision boundary** rather than overfitting to imbalance.

### 3. Dual Resampling Strategy
- Created **two** versions of the training set:
  - **Logistic Regression**: Resampled using **scaled features** (`X_train_scaled`).
  - **Random Forest**: Resampled using **unscaled features** (`X_train`), since tree-based models don’t require scaling.
- Both models were then trained on their respective resampled datasets.
- Printed class counts before and after to verify balance.

This step ensures that:
- The training data is balanced for better classification.
- No leakage occurs across datasets.
- Each model is optimized based on its assumptions (scaling vs no scaling).

## **Train & Tune Candidate Models**

We trained and optimized two classification models: **Logistic Regression** and **Random Forest**, using stratified cross-validation and class balancing. Our objective was to maximize **ROC AUC**, a robust metric for binary classification.

---

### Logistic Regression (with Regularization)

We trained multiple versions of Logistic Regression using:
- Solvers: `liblinear`, `lbfgs`, and `saga`
- Regularization types: `L1`, `L2`, and `ElasticNet`
- C values (inverse of regularization strength): from `0.001` to `1000`

**Why Stratified K-Fold Cross-Validation?**
- Our target variable (`is_popular`) is imbalanced (only ~10% of songs are popular).
- **Stratified K-Fold** ensures each fold maintains the same proportion of popular vs. non-popular songs.
- This prevents misleading validation results that could arise from uneven class distributions during cross-validation.

**Why ROC AUC?**
- Accuracy can be misleading with imbalanced classes — it may appear high even if the model fails to detect minority-class examples.
- **ROC AUC (Receiver Operating Characteristic - Area Under Curve)** measures the model’s ability to distinguish between classes across all classification thresholds.
- It's threshold-independent and highlights how well the model ranks true positives higher than false positives.

**Search Strategy:**
- Used `GridSearchCV` with 10-fold Stratified K-Fold
- Evaluated models using ROC AUC
- Trained on the **scaled, resampled training set** (`X_train_resampled_lr`)

**Results:**
- Best Solver: `liblinear`
- Best Parameters: `penalty='l1'`, `C=0.1`
- Best Cross-Validation AUC: **~0.80**

---

### Random Forest Classifier

We tuned a Random Forest using `RandomizedSearchCV` across:
- `n_estimators`: `[50, 100]`
- `max_depth`: `[5, 10, 20]`
- `min_samples_split`: `[2, 5, 10]`
- `max_features`: `['sqrt']`

**Search Strategy:**
- Used `RandomizedSearchCV` with 18 random combinations
- Applied 10-fold **Stratified K-Fold** cross-validation
- Scored using **ROC AUC**
- Trained on the **unscaled, resampled training set** (`X_train_resampled_rf`), since scaling is not required for tree-based models

**Results:**
- Best Parameters:
  - `n_estimators=100`
  - `max_depth=10`
  - `min_samples_split=5`
  - `max_features='sqrt'`
- Best Cross-Validation AUC: **~0.89**

- This Random Forest model outperformed Logistic Regression in ROC AUC and was selected as our primary model for threshold tuning and final test evaluation.

## **Threshold Tuning & Validation Evaluation**

After training and selecting the best-performing Logistic Regression and Random Forest models, we performed threshold tuning to optimize classification results on the validation set.

---

### Why Threshold Tuning?

Most classifiers (e.g., Logistic Regression, Random Forest) output probabilities, not binary predictions. By default, a threshold of 0.50 is used to classify a prediction as positive (e.g., “popular”).

However:
- With imbalanced datasets, this default often leads to suboptimal precision, recall, or F1 scores.
- We tuned thresholds to maximize F1 score, which balances precision and recall.

---

### What We Did

**Random Forest:**
- Collected predicted probabilities on the validation set.
- Evaluated model performance across thresholds from 0.10 to 0.85, tracking:
  - F1 Score
  - Precision
  - Recall
  - Accuracy
- Selected the threshold that produced the best F1 score.
- At that threshold, we reported:
  - Precision, Recall, F1 Score, Accuracy
  - Confusion Matrix
  - Classification Report

**Logistic Regression:**
- Repeated the same process using the scaled validation set.
- Evaluated at the same range of thresholds and used F1 optimization to choose the best cutoff.
- Reported detailed performance metrics and confusion matrix at the selected threshold.

---

### Visualization & Insights
- Plotted Threshold vs. Metric Curves for both models:
  - Allowed visual inspection of trade-offs between precision, recall, and F1 score.
- Generated Confusion Matrices at the optimized threshold for both models to visualize classification performance.

---

### Key Benefits of This Step:
- Avoids relying on arbitrary thresholds.
- Customizes the model’s classification behavior to the specific business goal (e.g., balance vs. precision).
- Validates model robustness across different cutoff levels, especially in imbalanced settings.

This tuning ensured that we were evaluating each model fairly and optimally, rather than relying on default decision boundaries that may not reflect true performance potential.

## **Final Model Evaluation on Test Set**

After tuning and validating our Random Forest model using threshold optimization on the validation set, we evaluated its performance on the **unseen test set** to measure how well the model generalizes.

---

### What We Did

1. **Applied the optimized threshold** (0.55) to the predicted probabilities generated by the Random Forest model.
2. **Computed key classification metrics** including:
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - Specificity
   - F1 Score
   - ROC AUC
3. **Visualized results** using:
   - A confusion matrix to show classification breakdown
   - An ROC curve to evaluate performance across all thresholds

---

### Final Random Forest Test Results (Threshold = 0.55)

- **Accuracy**: 0.8873  
  The overall percentage of correct predictions.
  
- **Recall (Sensitivity)**: 0.6241  
  The proportion of actual popular songs correctly identified by the model.
  
- **Specificity**: 0.9194  
  The proportion of non-popular songs correctly predicted as not popular.
  
- **Precision**: 0.4853  
  Of all the songs predicted to be popular, 48.5% were truly popular.
  
- **F1 Score**: 0.5460  
  A balanced metric combining precision and recall.
  
- **AUC**: 0.8857  
  The model’s ability to rank predictions correctly across all thresholds.

---

### Summary

The Random Forest model achieved **strong generalization performance** on the test set. It balanced **recall and precision** well given the class imbalance and showed **excellent ranking ability** with an AUC of 0.8857.

This performance confirms that the model is suitable for real-world use cases such as:
- Filtering likely breakout songs
- Recommending high-potential tracks
- Prioritizing songs for promotion or analysis

All final metrics and plots were computed without any exposure to test data during training or validation.

