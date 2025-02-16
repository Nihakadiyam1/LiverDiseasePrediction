from sklearn.model_selection import train_test_split

# Assuming X and y have been defined as the feature set and target variable respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Count the number of instances of each class in the training target 'y_train'
train_target_counts = y_train.value_counts()
print(train_target_counts)