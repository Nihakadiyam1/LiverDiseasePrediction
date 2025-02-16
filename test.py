import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
liver_data = pd.read_csv('Liver_disease_data.csv')

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    liver_data.drop('Diagnosis', axis=1),
    liver_data['Diagnosis'],
    test_size=0.2,
    random_state=42
)

# Print the shapes
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)