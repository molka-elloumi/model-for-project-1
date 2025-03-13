<<<<<<< HEAD
#start by understanding data 
data_set = pd.read_csv("heart_failure_clinical_records_dataset.csv")
=======
#libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#setting the important variables
data_set = pd.read_csv("heart_failure_clinical_records_dataset.csv")


# Check for missing values
print("\nMissing Values Per Column:\n")
print(data_set.isnull().sum())

# Visualize missing values using a heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data_set.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()
#no missing values
# Outlier Detection & Management

# Visualizing outliers using box plots
plt.figure(figsize=(12,6))
data_set.boxplot(rot=90) # Changed df to data_set
plt.title("Boxplot of Numerical Features")
plt.show()

# Identifying outliers using the IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Apply outlier detection to numerical columns
outliers = {}
for col in data_set.select_dtypes(include=[np.number]).columns: # Changed df to data_set
    outliers[col] = detect_outliers_iqr(data_set, col) # Changed df to data_set
    print(f"Outliers detected in {col}: {len(outliers[col])}")

# Handling Outliers - Capping
for col in data_set.select_dtypes(include=[np.number]).columns: # Changed df to data_set
    Q1 = data_set[col].quantile(0.25) # Changed df to data_set
    Q3 = data_set[col].quantile(0.75) # Changed df to data_set
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_set[col] = np.where(data_set[col] < lower_bound, lower_bound, data_set[col]) # Changed df to data_set
    data_set[col] = np.where(data_set[col] > upper_bound, upper_bound, data_set[col]) # Changed df to data_set

print("\nOutliers have been handled using capping.")
>>>>>>> 782e93c599ad3fc4b5cca12129f8b9d292609a77
