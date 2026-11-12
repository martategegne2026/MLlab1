import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SETUP & DATA CREATION
np.random.seed(42)
n = 100
study_hours = np.random.normal(5, 2, n)
attendance = np.random.normal(85, 10, n)
previous_mark = np.random.normal(70, 15, n)

final_score = (0.5 * study_hours) + (0.3 * attendance) + (0.2 * previous_mark) + np.random.normal(0, 5, n)

df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance': attendance,
    'previous_mark': previous_mark,
    'final_score': final_score
})

# Injecting dirty data 
df.loc[0:5, 'study_hours'] = np.nan  # Missing values
df.loc[99, 'study_hours'] = 50       # Outlier
df.loc[98, 'attendance'] = -10       # Impossible value

print("--- INITIAL DATA (First 5 rows) ---")
print(df.head())

# PART A — INSPECT AND CLEAN
# 1. Handle missing values & impossible values using median imputation
df['study_hours'] = df['study_hours'].fillna(df['study_hours'].median())
df.loc[df['attendance'] < 0, 'attendance'] = np.nan
df['attendance'] = df['attendance'].fillna(df['attendance'].median())

# 2. Cap the outlier in study_hours at 15
df['study_hours'] = np.where(df['study_hours'] > 15, 15, df['study_hours'])

# Feature Matrix Shape & Target definition
# Feature matrix (X) shape: 100 rows x 3 columns
X = df[['study_hours', 'attendance', 'previous_mark']]
# Target variable (y): final_score
y = df['final_score']
print(f"\nFeature Matrix Shape: {X.shape}")


# PART B — SUMMARIZE
print("\n--- SUMMARY STATISTICS ---")
summary_stats = df.agg(['mean', 'median', 'std'])
print(summary_stats)
# PART C — VISUALIZE
# 1. Histogram for final_score
plt.figure(figsize=(8, 5))
plt.hist(df['final_score'], bins=15, edgecolor='black', color='skyblue')
plt.title('Distribution of Final Scores')
plt.xlabel('Final Score')
plt.ylabel('Frequency')
plt.savefig('final_score_histogram.png') # Saves visualization to workspace
plt.close()

# 2. Scatter plot of study_hours vs final_score
plt.figure(figsize=(8, 5))
sns.scatterplot(x='study_hours', y='final_score', data=df, color='purple')
plt.title('Study Hours vs Final Score')
plt.xlabel('Study Hours')
plt.ylabel('Final Score')
plt.savefig('study_hours_vs_score.png') # Saves visualization to workspace
plt.close()
print("\nVisualizations saved successfully as PNG files.")


# PART D — CORRELATION
correlation = df['study_hours'].corr(df['final_score'])
print(f"\nPearson Correlation between study_hours and final_score: {correlation:.4f}")


# PART E — VECTORS AND DOT PRODUCT

# Select one student row as a vector x (excluding final_score)
x_vector = X.loc[10].to_numpy()
w_vector = np.array([0.5, 0.2, 0.4])
bias = 5

predicted_score = np.dot(w_vector, x_vector) + bias
print(f"\nVector Dot Product Calculation (Student 10): {predicted_score:.4f}")

# PART F — SCALING

df['study_hours_scaled'] = (df['study_hours'] - df['study_hours'].mean()) / df['study_hours'].std()
df['previous_mark_scaled'] = (df['previous_mark'] - df['previous_mark'].mean()) / df['previous_mark'].std()

print("\n--- SCALED DATA (First 5 rows) ---")
print(df[['study_hours_scaled', 'previous_mark_scaled']].head())

# Export cleaned dataset for records
df.to_csv('student_data_cleaned.csv', index=False)