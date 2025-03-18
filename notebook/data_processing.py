
'''###**Optimizing Memory Usage**
* **Objective:**

We aim to reduce the memory consumption of the DataFrame by optimizing data types. This will improve computational efficiency and scalability, especially for large datasets.

* **Approach:**
* Convert float64 to float32 to save memory.

* Convert int64 to int32 for integer columns.

* Convert object columns to category for columns with many repeated values.

These steps are applied using the optimize_memory() function'''

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def optimize_dtypes(df):
    """
    Convert float64 to float32 and int64 to int32 in a DataFrame.

    """
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df



print("Before Conversion:")
print(df.dtypes)

df = optimize_dtypes(df)

print("\nAfter Conversion:")
print(df.dtypes)
"""##OUTPUT:
Before Conversion:
age                         float64
anaemia                       int64
creatinine_phosphokinase      int64
diabetes                      int64
ejection_fraction             int64
high_blood_pressure           int64
platelets                   float64
serum_creatinine            float64
serum_sodium                  int64
sex                           int64
smoking                       int64
time                          int64
DEATH_EVENT                   int64
dtype: object

After Conversion:
age                         float32
anaemia                       int32
creatinine_phosphokinase      int32
diabetes                      int32
ejection_fraction             int32
high_blood_pressure           int32
platelets                   float32
serum_creatinine            float32
serum_sodium                  int32
sex                           int32
smoking                       int32
time                          int32
DEATH_EVENT                   int32
dtype: object"""

#**Memory Optimization Before and After**:

"""We’ll compare the memory usage of each column before and after optimization to see how much memory has been saved."""

# Define the function to optimize memory usage
def optimize_memory(df):
    """
    Optimizes the memory usage of a DataFrame by converting columns to more memory-efficient data types.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with optimized memory usage.
    """
    for col in df.columns:
        # Convert float64 to float32
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        # Convert int64 to int32
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
        # Convert object type (strings) to category if there are many repeated values
        elif df[col].dtype == object:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            # Convert to category if more than 10% of the column's values are unique
            if num_unique_values / num_total_values < 0.1:
                df[col] = df[col].astype('category')
    return df


# Memory usage before optimization
print("Before Optimization:")
before_memory = df.memory_usage(deep=True)
print(before_memory)

# Apply the memory optimization function
df_optimized = optimize_memory(df)

# Memory usage after optimization
print("\nAfter Optimization:")
after_memory = df_optimized.memory_usage(deep=True)
print(after_memory)

# Visualize the memory improvement before and after optimization
memory_comparison = pd.DataFrame({
    'Before Optimization': before_memory,
    'After Optimization': after_memory
})

# Plotting the memory comparison
memory_comparison.plot(kind='bar', figsize=(10, 6), title="Memory Usage Comparison (all columns) Before and After Optimization")
plt.ylabel('Memory Usage (bytes)')
plt.xlabel('Columns')
plt.xticks(rotation=45)
plt.show()
import matplotlib.pyplot as plt

# Create bar chart for memory comparison
plt.figure(figsize=(4, 3))
plt.bar(["Before Optimization", "After Optimization"], [initial_memory, optimized_memory], color=['red', 'green'])
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Before and After Optimization")
plt.ylim(0, initial_memory * 1.2)
plt.text(0, initial_memory, f"{initial_memory:.2f} MB", ha='center', va='bottom', fontsize=12)
plt.text(1, optimized_memory, f"{optimized_memory:.2f} MB", ha='center', va='bottom', fontsize=12)
plt.show()

"""Before Optimization:
Index                        132
age                         2392
anaemia                     2392
creatinine_phosphokinase    2392
diabetes                    2392
ejection_fraction           2392
high_blood_pressure         2392
platelets                   2392
serum_creatinine            2392
serum_sodium                2392
sex                         2392
smoking                     2392
time                        2392
DEATH_EVENT                 2392
dtype: int64

After Optimization:
Index                        132
age                         1196
anaemia                     1196
creatinine_phosphokinase    1196
diabetes                    1196
ejection_fraction           1196
high_blood_pressure         1196
platelets                   1196
serum_creatinine            1196
serum_sodium                1196
sex                         1196
smoking                     1196
time                        1196
DEATH_EVENT                 1196
dtype: int64"""

import matplotlib.pyplot as plt

# Create bar chart for memory comparison
plt.figure(figsize=(4, 3))
plt.bar(["Before Optimization", "After Optimization"], [initial_memory, optimized_memory], color=['red', 'green'])
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Before and After Optimization")
plt.ylim(0, initial_memory * 1.2)
plt.text(0, initial_memory, f"{initial_memory:.2f} MB", ha='center', va='bottom', fontsize=12)
plt.text(1, optimized_memory, f"{optimized_memory:.2f} MB", ha='center', va='bottom', fontsize=12)
plt.show()

"""
* **After optimization:**

* float64 to float32 conversion reduces memory for floating-point columns.

* int64 to int32 conversion helps save memory for integer columns.

* object to category conversion optimizes columns with repetitive strings.

This process results in a more efficient DataFrame, reducing memory usage and improving model performance."""