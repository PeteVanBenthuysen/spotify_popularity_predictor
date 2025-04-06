# **Spotify Song Popularity Prediction**

## **Goal**
The goal of this project is to **predict whether a song is popular** on Spotify, with popularity defined as a score greater than or equal to **70**. By understanding the factors that influence a song’s popularity, we aim to identify what variables cause a song to go mainstream, making this a valuable tool for artists, producers, and marketers who wish to optimize song appeal.

## **Why This Goal?**
Getting a song played on mainstream radio or reaching a large audience on streaming platforms is **not an easy task**. There are numerous factors—such as tempo, danceability, energy, and other audio features—that contribute to a song's success. By analyzing these features, we aim to build a model that can predict whether a song is likely to become popular, based on its characteristics.

## **Visualizations and Analyses:**
The following visualizations will be used to understand the dataset:

### **Histograms:**
- **Tempo** – Distribution of song tempo across the dataset.
- **Energy** – Distribution of energy levels in songs.
- **Danceability** – Distribution of danceability scores.
- **Popularity** – Distribution of the popularity scores of songs.

### **Bar Plot:**
- **Song x Counts per Key** – Displaying how songs are distributed across different musical keys.

### **Scatterplots:**
- **Danceability vs Popularity** – Examining the relationship between danceability and popularity scores.

### **Group by Comparisons:**
- **Is Popular vs Energy** – Comparing the energy levels between popular and non-popular songs to determine any significant differences.

### **Additional EDA Ideas:**
- **Correlation Matrix**: Analyze correlations between features to understand how different variables are related. This can help identify which features are most important for the model.
- **Box Plots**: For identifying potential outliers in variables like tempo, energy, danceability, and popularity.
- **Missing Data Analysis**: Investigate any missing values in the dataset to decide whether to fill them with mean/median or drop rows/columns.

**Note**: If needed, additional preprocessing steps (like scaling or normalizing certain features) will be done based on the EDA findings.

## **Train/Tune Candidate Models**

### **Modeling Approach:**
We will use the following machine learning models for training and tuning to predict the popularity of songs:

#### **K-Nearest Neighbors (K-NN):**
K-NN is a simple, yet effective model for classification problems. We will tune the `k` parameter and evaluate performance using metrics like **accuracy**, **F1-score**, and **confusion matrix**.

#### **Logistic Regression with Regularization:**
Logistic Regression is an ideal model for binary classification tasks. We will apply **regularization** (L1 or L2) to prevent overfitting and improve generalization. This model will predict whether a song's popularity is above or below **70** (binary classification).

Both models will be tuned using **cross-validation** and evaluated on the **test set** to assess their accuracy in predicting song popularity.
