import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Combine "Movie" and "Soundtrack" into "Movie_Soundtrack"
df['genre'] = df['genre'].replace({'Movie': 'Movie_Soundtrack', 'Soundtrack': 'Movie_Soundtrack'})

# Combine "Hip-Hop" and "Rap" into "Hip-Hop_Rap"
df['genre'] = df['genre'].replace({'Hip-Hop': 'Hip-Hop_Rap', 'Rap': 'Hip-Hop_Rap'})

# Check the updated genre distribution
print(df['genre'].value_counts())

# Drop columns we don't want the model to use
df = df.drop(columns=['track_name', 'track_id', 'artist_name', 'duration_ms'])

# Statistically define threshold for popularity based on top 10%
threshold = df['popularity'].quantile(0.9)  # top 10%
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

#### 2) EDA ###

## WE WILL STILL BE REMOVING DATA POINTS HERE DUE TO OUTLIERS, ETC REMOVE MSG WHEN DONE ###

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df['duration_sec'], bins=100, edgecolor='black')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Song Duration')
plt.xlim(0, 600)  # Focus on songs between 0 and 10 minutes (600 seconds)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Save original count before filtering
original_count = df.shape[0]

# IQR calculation
Q1 = df['duration_sec'].quantile(0.25)
Q3 = df['duration_sec'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR  # ✅ You were missing this line

# Filter using IQR
df = df[(df['duration_sec'] >= lower_bound) & (df['duration_sec'] <= upper_bound)]

# Count after filtering
filtered_count = df.shape[0]

# Show how many rows were removed
removed_count = original_count - filtered_count
print(f"Removed {removed_count} songs using IQR filtering.")
print(f"IQR Range: {lower_bound:.0f} to {upper_bound:.0f}")
print(f"Duration range after filtering: {df['duration_sec'].min():.3f} to {df['duration_sec'].max():.3f}")


## DUMMY CODING AND ALL OF 4 NEEDS TO BE DONE AFTER SPLITTING THE DATASET ##

### 4) Feature Engineering ###
# Convert mode from string to binary: Major = 1, Minor = 0
## df['mode'] = df['mode'].map({'Minor': 0, 'Major': 1})

# One-hot encode categorical variables
## df = pd.get_dummies(df, columns=['genre', 'key'])