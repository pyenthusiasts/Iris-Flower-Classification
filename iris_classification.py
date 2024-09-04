# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()

# Features (X) and target (y)
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the target values for the testing data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Predict the species of a new sample
new_sample = [[5.0, 3.6, 1.4, 0.2]]  # Example features: sepal length, sepal width, petal length, petal width
prediction = clf.predict(new_sample)
print("\nPredicted Species:", iris.target_names[prediction[0]])
