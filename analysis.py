import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model selection and evaluation
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Preprocessing
from sklearn.preprocessing import StandardScaler

### 1) Loading, cleaning, and preprocessing the dataset ###

# Load dataset
df = pd.read_csv("data/SpotifyFeatures.csv", parse_dates=False) # Set parse_dates=False to avoid parsing dates
print(df.head())

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
print(df['genre'].value_counts())

# Drop columns we don't want the model to use
df = df.drop(columns=['track_name', 'track_id', 'artist_name', 'duration_ms'])

# Statistically define threshold for popularity based on top 10%
threshold = df['popularity'].quantile(0.90)  # top 10%
df['is_popular'] = (df['popularity'] >= threshold).astype(int)
print(f"Using popularity threshold: {threshold:.2f}")
print(df['is_popular'].value_counts(normalize=True))  # Check class distribution

df = df.drop(columns=['popularity'])
df = df.drop_duplicates()

# Final quality checks
print(f"Missing values:\n{df.isnull().sum()}\n")       # Check for missing values
print(f"\nData types:\n{df.dtypes}\n")                 # Confirm correct data types
print(f"Columns in dataset: {list(df.columns)}")       # Final column names

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
print(f"Removed {removed_count} songs using IQR filtering.")
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
print(f"Removed {removed_count} songs using IQR filtering.")
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

### 6) Train/Tune Candidate Models ###

# The parameter grid for Logistic Regression with regularization
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l2'],  # L2 regularization
    'solver': ['lbfgs'],  # Solver that supports L2 regularization
}

# Initialize GridSearchCV with Logistic Regression and 5-fold cross-validation
grid_lr = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    param_grid=param_grid_lr,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Evaluate using accuracy
    n_jobs=-1  # Use all available CPU cores for faster processing
)

# Fit the model on the training data (scaled data for Logistic Regression)
grid_lr.fit(X_train_scaled, y_train)

# Output the best hyperparameters and best score
print("Best parameters for Logistic Regression:", grid_lr.best_params_)
print("Best cross-validation accuracy for Logistic Regression:", grid_lr.best_score_)

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 5, 10, 20],  # Max depth of each tree
    'min_samples_split': [2, 5, 10]  # Minimum samples required to split a node
}

# Initialize GridSearchCV with Random Forest and 5-fold cross-validation
grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid_rf,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Evaluate using accuracy
    n_jobs=-1  # Use all available CPU cores
)

# Fit the model on the training data (no scaling needed for Random Forest)
grid_rf.fit(X_train, y_train)

# Output the best hyperparameters and best score
print("Best parameters for Random Forest:", grid_rf.best_params_)
print("Best cross-validation accuracy for Random Forest:", grid_rf.best_score_)

### 6.5) Evaluating Tuned Models on the Validation Set ###

# Get the best models from GridSearchCV
best_logreg = grid_lr.best_estimator_
best_rf = grid_rf.best_estimator_

# Make predictions on the validation set
y_val_pred_lr = best_logreg.predict(X_val_scaled)
y_val_pred_rf = best_rf.predict(X_val)

print("Logistic Regression - Validation Set Performance:")
print(classification_report(y_val, y_val_pred_lr))

print("\nRandom Forest - Validation Set Performance:")
print(classification_report(y_val, y_val_pred_rf))

# THRESHOLD TUNING (LOGISTIC REGRESSION)

# Get predicted probabilities
y_val_probs_lr = best_logreg.predict_proba(X_val_scaled)[:, 1]

# Set custom threshold
threshold = 0.3
y_val_pred_lr_thresh = (y_val_probs_lr >= threshold).astype(int)

# Print adjusted performance
print(f"\nLogistic Regression - Validation Set (Threshold = {threshold}):")
print(classification_report(y_val, y_val_pred_lr_thresh))

# Function to plot a confusion matrix
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

# Plot confusion matrix for Logistic Regression
plot_conf_matrix(y_val, y_val_pred_lr, 'Logistic Regression - Validation Set')

# Plot confusion matrix for Random Forest
plot_conf_matrix(y_val, y_val_pred_rf, 'Random Forest - Validation Set')