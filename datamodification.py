import pandas as pd

df = pd.read_csv('dataset/diabetes_dataset_v0.csv')
y = df['target']
df['BMI_cube'] = df['bmi'] ** 3
df.to_csv('dataset/diabetes_dataset_v0.csv', index=False)


