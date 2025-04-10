# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model selection & evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split as tts

# Preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


### 1) Loading, cleaning, and preprocessing the dataset ###

# Load dataset
df = pd.read_csv("data/SpotifyFeatures.csv", parse_dates=False) # Set parse_dates=False to avoid parsing dates
print("First 5 Rows of the Dataset:")
print(df.head())

print("Count of Missing Values in Each Column:")
print(df.isnull().sum()) # Check for missing values

# Check for duplicate track_ids with different genres
dupes = df[df.duplicated(subset=['track_id'], keep=False)]
num_dupes = dupes['track_id'].nunique()
total_dupe_rows = dupes.shape[0]

print(f"\nFound {num_dupes} duplicated track_ids with conflicting genres.")
print(f"This affects a total of {total_dupe_rows} rows.\n")
print("Sample of duplicate conflicts:")
print(dupes[['track_id', 'track_name', 'genre']].sort_values(by='track_id').head(10))

before_dedup = df.shape[0]
# Remove duplicates based on track_id, keeping the first occurrence
df = df.drop_duplicates(subset='track_id', keep='first')
after_dedup = df.shape[0]

print(f"\nRemoved {before_dedup - after_dedup} duplicate rows based on track_id.\n")

# Checking output of time_signature
print("Proportion of Each Value in 'time_signature':")
print(df['time_signature'].value_counts(normalize=True))

# Remove rows with invalid time signature and convert to integer
df = df[df['time_signature'] != '0/4']
df['time_signature'] = df['time_signature'].str.extract('(\d+)').astype(int)

# Convert duration from milliseconds to seconds for easier interpretation
df['duration_sec'] = df['duration_ms'] / 1000

# Fixing the genre name
df['genre'] = df['genre'].replace("Children’s Music", "Children's Music") 

# Dropping genres that are not relevant for our analysis
df = df[~df['genre'].isin(['Comedy', "Children's Music"])]

# Combine "Hip-Hop" and "Rap" into "Hip-Hop_Rap"
df['genre'] = df['genre'].replace({'Hip-Hop': 'Hip-Hop_Rap', 'Rap': 'Hip-Hop_Rap'})

# Check the updated genre distribution
print("Count of Each Genre in the Dataset:")
print(df['genre'].value_counts())

# Drop columns we don't want the model to use
df = df.drop(columns=['track_name', 'track_id', 'artist_name', 'duration_ms'])

# Statistically define threshold for popularity based on top 10%
threshold = df['popularity'].quantile(0.90) # Top 10%
df['is_popular'] = (df['popularity'] >= threshold).astype(int)
print(f"Using popularity threshold: {threshold:.2f}")
print("Class Distribution for 'is_popular' (Proportions):")
print(df['is_popular'].value_counts(normalize=True)) # Check class distribution



df = df.drop(columns=['popularity'])
df = df.drop_duplicates()

# Final quality checks
print("\n Quality checks:")
print("Missing values:\n")
print(df.isnull().sum())       # Check for missing values

print("\nData types:\n")
print(df.dtypes)               # Confirm correct data types

print("\nColumns in dataset:")
print(list(df.columns))        # Final column names

# Final duplicate check
num_duplicates = df.duplicated(keep=False).sum()
print(f"\nExact duplicate rows: {num_duplicates}")

if num_duplicates > 0:
    print("Sample duplicates:\n", df[df.duplicated(keep=False)].head(10))
else:
    print("No exact duplicate rows found.")

print(f"Cleaned dataset shape: {df.shape}")

### 1.5) EDA Checks ###

print(f"\nSummary statistics:\n{df.describe()}")  # Summary statistics
print(f"\nTime signature distribution:\n{df['time_signature'].value_counts(normalize=True)}")  # Categorical breakdown

### 2) EDA ###

# Genre overview via bar chart
genre_counts = df['genre'].value_counts() # Count songs per genre
 
# Plot
plt.figure(figsize=(12, 8))
genre_counts.plot(kind='barh', color='skyblue', edgecolor='black')

# Add titles and labels
plt.title('Genre Distribution (All Genres)', fontsize=16)
plt.xlabel('Number of Songs', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Flip the y-axis to show most common genres on top
plt.gca().invert_yaxis()

# Display the plot
plt.tight_layout()
plt.show()

# Save original count before filtering
original_count = df.shape[0]

# IQR calculation
Q1 = df['duration_sec'].quantile(0.25)
Q3 = df['duration_sec'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR 

# Filter using IQR
df = df[(df['duration_sec'] >= lower_bound) & (df['duration_sec'] <= upper_bound)]

# Count after filtering
filtered_count = df.shape[0]

# Show how many rows were removed
removed_count = original_count - filtered_count
print(f"Removed {removed_count} songs due to length outliers using IQR filtering.")
print(f"IQR Range: {lower_bound:.0f} to {upper_bound:.0f}")
print(f"Duration range after filtering: {df['duration_sec'].min():.3f} to {df['duration_sec'].max():.3f}")

# Plot boxplot of song tempo by genre
sns.boxplot(x='genre', y='tempo', data=df)
plt.title('Tempo Distribution by Genre')
plt.xticks(rotation=45)
plt.show()

# Plot histogram of song tempo
plt.figure(figsize=(10, 6))
plt.hist(df['tempo'], bins=100, edgecolor='black')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Frequency')
plt.title('Distribution of Song Tempo')
plt.xlim(0, 250)  # Focus on songs between 0 and 250 BPM (common tempo range)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Save original count before filtering
original_count = df.shape[0]

# IQR calculation for tempo
Q1 = df['tempo'].quantile(0.25)
Q3 = df['tempo'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR 

# Filter using IQR for tempo
df = df[(df['tempo'] >= lower_bound) & (df['tempo'] <= upper_bound)]

# Count after filtering
filtered_count = df.shape[0]

# Show how many rows were removed
removed_count = original_count - filtered_count
print(f"Removed {removed_count} songs due to tempo outliers using IQR filtering.")
print(f"IQR Range: {lower_bound:.0f} to {upper_bound:.0f}")
print(f"Tempo range after filtering: {df['tempo'].min():.3f} to {df['tempo'].max():.3f}")

# Violin plot for is_popular vs energy
sns.violinplot(x='is_popular', y='energy', data=df)
plt.title('Energy Distribution by Top 10% Popularity (is_popular)')
plt.xlabel('Is Popular (1 = Yes, 0 = No)')
plt.ylabel('Energy')
plt.show()

# Calculate IQR for energy based on is_popular
print("IQR (Popular):", df[df['is_popular'] == 1]['energy'].quantile(0.75) - df[df['is_popular'] == 1]['energy'].quantile(0.25))
print("IQR (Not Popular):", df[df['is_popular'] == 0]['energy'].quantile(0.75) - df[df['is_popular'] == 0]['energy'].quantile(0.25))

# Scatter plot for loudness vs. danceability
x = 'danceability'  # Defined variables
y = 'loudness'

corr_coeff = np.corrcoef(df[x], df[y])[0, 1] # Calculate the correlation coefficient

# Print summary statistics for both features
print(f"Summary Statistics for {x.capitalize()}:")
print(df[x].describe())
print(f"\nSummary Statistics for {y.capitalize()}:")
print(df[y].describe())

# Plot the scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=df)

# Annotate the plot with the correlation coefficient
plt.title(f'{x.capitalize()} vs. {y.capitalize()} \nCorrelation: {corr_coeff:.2f}')
plt.xlabel(x.capitalize())
plt.ylabel(y.capitalize())
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# Scatter plot for loudness vs energy
x = 'loudness' # Defined variables
y = 'energy'

# Calculate the correlation coefficient
corr_coeff = np.corrcoef(df[x], df[y])[0, 1]

# Print summary statistics for both features
print(f"Summary Statistics for {x.capitalize()}:")
print(df[x].describe())
print(f"\nSummary Statistics for {y.capitalize()}:")
print(df[y].describe())

# Plot the scatter plot of loudness vs energy
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=df)

# Annotate the plot with the correlation coefficient
plt.title(f'{x.capitalize()} vs. {y.capitalize()} \nCorrelation: {corr_coeff:.2f}')
plt.xlabel(x.capitalize())
plt.ylabel(y.capitalize())
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# Print the correlation coefficient
print(f"\nCorrelation between {x} and {y}: {corr_coeff:.2f}")

# Calculate pairwise correlation of features
corr_matrix = df[['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Feature Interaction Correlation Matrix')
plt.show()

# Data point check
print(f"After EDA and cleaning, the final dataset contains {df.shape[0]:,} songs.")

### 3) Training/Testing Dataset Using Train/Validation/Test For Baseline Model ###

# First split: 80% train_val, 20% test
train_val_df, test_df = tts(
    df, test_size=0.2, stratify=df['is_popular'], random_state=42
)

# Second split: 75% train, 25% val from train_val
train_df, val_df = tts(
    train_val_df, test_size=0.25, stratify=train_val_df['is_popular'], random_state=42
)

# Define feature columns (drop the target)
feature_cols = train_df.select_dtypes(include='number').columns.drop('is_popular')

# X and y for training set
X_train = train_df[feature_cols]
y_train = train_df['is_popular']

# Validation set
X_val = val_df[feature_cols]
y_val = val_df['is_popular']

# Test set
X_test = test_df[feature_cols]
y_test = test_df['is_popular']

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

### 3.5) Baseline Model With Linear Regression ###

# Initialize and train model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Predict on validation set
y_val_pred = linreg.predict(X_val)

# Evaluate model performance
print("Linear Regression - R² Score:", r2_score(y_val, y_val_pred))
print("Linear Regression - MSE:", mean_squared_error(y_val, y_val_pred))

### 4) Feature Engineering ###

# Convert mode from string to binary: Major = 1, Minor = 0
df['mode'] = df['mode'].map({'Minor': 0, 'Major': 1})

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['genre', 'key'])

# Create a new feature for vocalness as the inverse of instrumentalness (more vocals = less instrumental)
df['vocalness'] = 1 - df['instrumentalness']

# Bin tempo into categories: slow, medium, fast
df['tempo_category'] = pd.cut(
    df['tempo'],
    bins=[0, 90, 120, float('inf')],
    labels=['slow', 'medium', 'fast']
)

# One-hot encode tempo_category
df = pd.get_dummies(df, columns=['tempo_category'])

# Calculate correlation matrix including the target variable
corr_matrix = df[['is_popular', 'danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                  'vocalness']].corr()
                  
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix Including is_popular')
plt.show()

# Drop the original instrumentalness column
df.drop(columns=['instrumentalness'], inplace=True)

# Drop the original tempo column to avoid redundancy
df.drop(columns=['tempo'], inplace=True)

# Data checkpoint after feature engineering
print("Checking for missing values in the dataset...") # Check for missing values in each column
missing_values = df.isnull().sum()
print("Missing values by column:")
print(missing_values) 
print("Checking for duplicate rows in the dataset...") # Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"After feature engineering, the dataset contains {df.shape[0]:,} songs.")

### 4.5) Setting Up New Train/Validation/Test Sets After Feature Engineering ###

# First split: 80% train_val, 20% test
train_val_df, test_df = tts(
    df, test_size=0.2, stratify=df['is_popular'], random_state=42
)

# Second split: 75% train, 25% val from train_val
train_df, val_df = tts(
    train_val_df, test_size=0.25, stratify=train_val_df['is_popular'], random_state=42
)

# Define feature columns (drop the target)
feature_cols = train_df.select_dtypes(include='number').columns.drop('is_popular')

# X and y for training set
X_train = train_df[feature_cols]
y_train = train_df['is_popular']

# Validation set
X_val = val_df[feature_cols]
y_val = val_df['is_popular']

# Test set
X_test = test_df[feature_cols]
y_test = test_df['is_popular']

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

### 5) Rescaling Training Data ###

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler only on the training data
scaler.fit(X_train)

# Transform the training data using the fitted scaler
X_train_scaled = scaler.transform(X_train)

# Transform the validation and test data using the same scaler (no fitting, just transforming)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Print the first few rows of the scaled data to ensure it looks correct
print("Scaled Training Data (first few rows):")
print(X_train_scaled[:5])

# Create separate SMOTE objects for each model to avoid overwriting
sm = SMOTE(random_state=42)

# Resample scaled data for Logistic Regression
X_train_resampled_lr, y_train_resampled_lr = sm.fit_resample(X_train_scaled, y_train)

# Resample unscaled data for Random Forest
X_train_resampled_rf, y_train_resampled_rf = sm.fit_resample(X_train, y_train)

print("Before oversampling:", np.bincount(y_train))
print("After oversampling for Logistic Regression:", np.bincount(y_train_resampled_lr))
print("After oversampling for Random Forest:", np.bincount(y_train_resampled_rf))

### 6) Train/Tune Candidate Models ###

# Logistic Regression Model

# Define multiple Logistic Regression solver configurations
logreg_models = [
    {
        'model': LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'),
        'params': {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100]
        }
    },
    {
        'model': LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced'),
        'params': [
            {
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1, 10, 100]
            },
            {
                'penalty': ['elasticnet'],
                'C': [0.01, 0.1, 1, 10, 100],
                'l1_ratio': [0.0, 0.5, 1.0]  # Only for elasticnet
            }
        ]
    },
    {
        'model': LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'),
        'params': {
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10, 100]
        }
    }
]

# Loop through each configuration and run GridSearchCV
best_score = 0
best_model = None
best_params = None

for config in logreg_models:
    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train_resampled_lr, y_train_resampled_lr)
    
    print(f"Solver: {config['model'].solver}")
    print("Best Params:", grid.best_params_)
    print("Best CV F1 Score:", grid.best_score_)
    print("-" * 40)
    
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_params = grid.best_params_

# Save the best performing model
best_logreg = best_model

print("Best Logistic Regression Model (Key Parameters):")
print(best_logreg)

#Random Forest Model
# Define the parameter distributions for Random Forest
param_dist_rf = {
    'n_estimators': [50, 100, 200],         # Number of trees
    'max_depth': [None, 5, 10, 20],         # Depth of trees
    'min_samples_split': [2, 5, 10],        # Min samples to split
    'max_features': ['sqrt',] #'log2', None]  # Feature subset at each split
}

# Initialize RandomizedSearchCV with Random Forest
random_rf = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist_rf,
    n_iter=25,                  # Try 25 random combinations
    cv=5,                       # 5-fold cross-validation
    scoring='f1',               # F1-score for binary classification
    n_jobs=-1,                  # Use all available CPU cores
    verbose=1,                  # Show progress
    random_state=42             # Reproducibility
)

# Fit the model on the training data (no scaling needed for Random Forest)
random_rf.fit(X_train_resampled_rf, y_train_resampled_rf)

# Output the best hyperparameters and best cross-validation score
print("Best parameters for Random Forest:", random_rf.best_params_)
print("Best cross-validation F1 score for Random Forest:", random_rf.best_score_)

# Save the best model for future use
best_rf = random_rf.best_estimator_

print("Best Random Forest Model (Key Parameters):")
print(best_rf)
### 6.5) Evaluating Tuned Models on the Validation Set ###

# Making predictions on the validation set

# Function to plot threshold metrics
def plot_threshold_metrics(thresholds, f1_scores, precisions, recalls, accuracies, title='Metric Performance vs. Classification Threshold'):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', marker='o')
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='o')
    plt.plot(thresholds, accuracies, label='Accuracy', marker='o')
    plt.axvline(0.5, color='gray', linestyle='--', label='Default Threshold (0.5)')
    plt.title(title)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Random Forest: Threshold Tuning

# Get predicted probabilities from the random forest model
y_val_probs_rf = best_rf.predict_proba(X_val)[:, 1]

# Store metrics for different thresholds
thresholds_rf = np.arange(0.1, 0.9, 0.05)
f1_scores_rf, precisions_rf, recalls_rf, accuracies_rf = [], [], [], []

for thresh in thresholds_rf:
    y_pred_thresh_rf = (y_val_probs_rf >= thresh).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_thresh_rf, average='binary')
    acc = accuracy_score(y_val, y_pred_thresh_rf)

    precisions_rf.append(precision)
    recalls_rf.append(recall)
    f1_scores_rf.append(f1)
    accuracies_rf.append(acc)

# Find best threshold based on F1 score
best_idx_rf = np.argmax(f1_scores_rf)
best_threshold_rf = thresholds_rf[best_idx_rf]
print(f"\nBest threshold for Random Forest (by F1-score): {best_threshold_rf:.2f}")
print(f"Precision: {precisions_rf[best_idx_rf]:.2f}, Recall: {recalls_rf[best_idx_rf]:.2f}, F1: {f1_scores_rf[best_idx_rf]:.2f}, Accuracy: {accuracies_rf[best_idx_rf]:.2f}")

# Make final predictions using the best threshold
y_val_pred_rf_thresh = (y_val_probs_rf >= best_threshold_rf).astype(int)

# Final Random Forest Validation Performance
print("\nRandom Forest - Validation Set Performance (Threshold = {:.2f}):".format(best_threshold_rf))
print(classification_report(y_val, y_val_pred_rf_thresh))

# Plot F1, precision, recall, and accuracy vs. threshold
plot_threshold_metrics(
    thresholds_rf,
    f1_scores_rf,
    precisions_rf,
    recalls_rf,
    accuracies_rf,
    title='Threshold Performance - Random Forest (Validation Set)'
)

# Logistic Regression: Threshold Tuning

# Get predicted probabilities from logistic regression model
y_val_probs = best_logreg.predict_proba(X_val_scaled)[:, 1]

# Store metrics for different thresholds
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores, precisions, recalls, accuracies = [], [], [], []

for thresh in thresholds:
    y_pred_thresh = (y_val_probs >= thresh).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_thresh, average='binary')
    acc = accuracy_score(y_val, y_pred_thresh)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    accuracies.append(acc)

# Find best threshold based on F1 score
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"\nBest threshold (by F1-score): {best_threshold:.2f}")
print(f"Precision: {precisions[best_idx]:.2f}, Recall: {recalls[best_idx]:.2f}, F1: {f1_scores[best_idx]:.2f}, Accuracy: {accuracies[best_idx]:.2f}")

# Make final predictions using the best threshold
y_val_pred_lr = (y_val_probs >= best_threshold).astype(int)

# Final Logistic Regression Validation Performance
print("\nLogistic Regression - Validation Set Performance (Threshold = {:.2f}):".format(best_threshold))
print(classification_report(y_val, y_val_pred_lr))

# Plot F1, precision, recall, and accuracy vs. threshold
plot_threshold_metrics(
    thresholds,
    f1_scores,
    precisions,
    recalls,
    accuracies,
    title='Threshold Performance - Logistic Regression (Validation Set)'
)

# Function to plot the confusion matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Popular', 'Popular'],
                yticklabels=['Not Popular', 'Popular'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Confusion matrices at optimized thresholds
plot_conf_matrix(
    y_val,
    y_val_pred_lr,
    f"Confusion Matrix - Logistic Regression (Validation Set, Threshold = {best_threshold:.2f})"
)

plot_conf_matrix(
    y_val,
    y_val_pred_rf_thresh,
    f"Confusion Matrix - Random Forest (Validation Set, Threshold = {best_threshold_rf:.2f})"
)

### 7) Characterizing random forest model performance on testing set ###

# 1. Predict probabilities and apply threshold
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf >= best_threshold_rf).astype(int)

# 2. Evaluation metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Confusion matrix + specificity
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
specificity_rf = tn_rf / (tn_rf + fp_rf)

# Print results
print(f"Random Forest - Test Set (Threshold = {best_threshold_rf:.2f}):")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Sensitivity (Recall): {recall_rf:.4f}")
print(f"Specificity: {specificity_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")

# Plot confusion matrix
plot_conf_matrix(
    y_test,
    y_pred_rf,
    f"Confusion Matrix - Random Forest (Test Set, Threshold = {best_threshold_rf:.2f})"
)

# ROC Curve for Random Forest
def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='darkgreen')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot it
plot_roc_curve(
    y_test,
    y_prob_rf,
    title=f'ROC Curve - Random Forest (Test Set)'
)