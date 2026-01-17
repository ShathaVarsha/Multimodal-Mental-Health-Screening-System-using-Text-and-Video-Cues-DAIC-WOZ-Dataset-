import pandas as pd

# Load the engineered features
DATA_PATH = 'outputs/engineered_features.csv'
df = pd.read_csv(DATA_PATH)

# Check depression_binary and phq8_score columns
if 'depression_binary' in df.columns:
    y = df['depression_binary']
    print('Unique values in depression_binary:', y.unique())
    print('Value counts:')
    print(y.value_counts())
else:
    y = df['phq8_score']
    print('Unique values in phq8_score:', y.unique())
    print('Value counts:')
    print(y.value_counts())
