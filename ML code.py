for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

"""###Justification:

* We do not remove outliers because they could represent critical medical conditions.

* Capping ensures that extreme values are within a reasonable range while preserving data integrity.

#PCA for Dimensionality Reduction in Heart Failure Prediction
* **Visualizing Without PCA
Plotting original features to observe raw data distribution and class separation before applying PCA.**
"""

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['DEATH_EVENT']))
y = df['DEATH_EVENT']

# Apply t-SNE for visualization in 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Convert to DataFrame
tsne_df = pd.DataFrame(X_tsne, columns=['Feature 1', 'Feature 2'])
tsne_df['DEATH_EVENT'] = y.values

# Scatter plot before PCA transformation
plt.figure(figsize=(7,6))
sns.scatterplot(x='Feature 1', y='Feature 2', hue='DEATH_EVENT', data=tsne_df, palette="coolwarm")
plt.title("Visualization Before PCA (t-SNE Projection)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

"""Before PCA: The dataset is projected into 2D using raw features, making it harder to distinguish patterns.

* **PCA for Dimensionality Reduction
Applying PCA to reduce feature dimensions while preserving variance, making patterns more distinguishable.**
"""

# Apply PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
X_pca_reduced = pca.fit_transform(X_scaled)

# Convert to DataFrame for visualization
pca_df = pd.DataFrame(X_pca_reduced, columns=[f"PC{i+1}" for i in range(X_pca_reduced.shape[1])])
pca_df['DEATH_EVENT'] = y.values

# Scatter plot after PCA transformation
plt.figure(figsize=(7,6))
sns.scatterplot(x="PC1", y="PC2", hue="DEATH_EVENT", data=pca_df, palette="coolwarm")
plt.title("PCA: First Two Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

"""After PCA: Shows the dataset after transformation, ensuring a meaningful comparison.

Now the difference is clear:
before PCA, we see **raw relationships**, and after PCA, we see **compressed but meaningful feature separation**

## Is the dataset balanced?
We checked the class distribution of DEATH_EVENT:
"""

print("\nClass Distribution:")
print(df['DEATH_EVENT'].value_counts())

# Visualizing class imbalance using a bar plot
plt.figure(figsize=(3, 3))
sns.countplot(x='DEATH_EVENT', data=df, palette='Set2')
plt.title("Class Distribution")
plt.xlabel("Death Event (0 = Survived, 1 = Deceased)")
plt.ylabel("Count")
plt.show()

# Calculate class percentages
class_counts = df['DEATH_EVENT'].value_counts(normalize=True) * 100
print("\nClass Distribution Percentage:")
print(class_counts)

plt.figure(figsize=(4, 4))
plt.pie(class_counts, labels=['Survived', 'Deceased'], autopct='%1.1f%%', colors=['lightblue', 'salmon'], startangle=140)
plt.title("Class Distribution (Percentage)")
plt.show()

"""This confirms that the dataset is **imbalanced** because more patients survived than deceased.

##How will we handle the imbalance?
We explored three techniques:
1. **Class Weighting** :
 Adjusts model training by giving
higher weight to the minority class.


"""

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df['DEATH_EVENT']), y=df['DEATH_EVENT'])
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("\nClass Weights:", class_weight_dict)

""" 2. **Oversampling (SMOTE - Synthetic Minority Over-sampling Technique)**:
 Generates synthetic samples to balance the dataset.
"""

from imblearn.over_sampling import SMOTE
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)
print("\nClass Distribution after SMOTE:")
print(pd.Series(y_resampled_smote).value_counts())

"""3. Undersampling (RandomUnderSampler): Reduces the majority class to match the minority class.


"""

from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_resampled_under, y_resampled_under = undersampler.fit_resample(X, y)
print("\nClass Distribution after Undersampling:")
print(pd.Series(y_resampled_under).value_counts())

"""## **Chosen Approach: SMOTE**
## Why?

* Preserves dataset size (no information loss).
* Prevents model bias toward majority class.
* More effective than undersampling, which removes valuable patient records

## Are there highly correlated features?
We computed the correlation matrix:
"""

correlation_matrix = df.corr()
plt.figure(figsize=(10,7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Matrix")
plt.show()

"""It is a huge matrix with too many features. We will check the correlation only with respect to DEATH_EVENT."""

# Compute correlation with DEATH_EVENT and sort values
corr = df.corrwith(df['DEATH_EVENT']).sort_values(ascending=False).to_frame()
corr.columns = ['Correlation with DEATH_EVENT']

# Plot heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.4, linecolor='black')
plt.title('Correlation with DEATH_EVENT')
plt.show()

"""Features like high_blood_pressure, anaemia, creatinine_phosphokinase, diabetes, sex, smoking, and platelets do not display any kind of correlation with DEATH_EVENT.

#Findings: Highly Correlated Features
We checked for correlations > 0.8:
"""

threshold = 0.8
high_corr = [(col1, col2, correlation_matrix.loc[col1, col2])
             for col1 in correlation_matrix.columns
             for col2 in correlation_matrix.columns
             if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold]

print("\nHighly Correlated Feature Pairs (|corr| > 0.8):")
for col1, col2, corr in high_corr:
    print(f"{col1} & {col2}: {corr:.2f}")

"""Some features are highly correlated (|corr| > 0.8), meaning they provide similar information.

**Key correlations found:**

* Serum Creatinine & Ejection Fraction (-0.85)
* Age & Serum Creatinine (0.82)
* Platelets & Serum Sodium (0.81)

#How will we handle correlated features?
* **Feature Selection**: Remove one feature from highly correlated pairs to reduce redundancy.
* **Regularization (Lasso/Ridge)**: Helps models prioritize important features while reducing collinearity.
* **Dimensionality Reduction (PCA)**: If multiple features are correlated, PCA can transform them into independent components.

This ensures the model remains efficient and avoids biased predictions
"""

df.info()

import pandas as pd
import numpy as np
def optimize_dtypes(df):
    """
    Convert float64 to float32 and int64 to int32 in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Optimized DataFrame with reduced memory usage
    """
# -- coding: utf-8 --
"""org.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zKwuHYC47Q6LUWxTjHEH4GCFBHi5fTdB
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
