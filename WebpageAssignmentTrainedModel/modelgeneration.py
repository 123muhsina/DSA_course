import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Read the dataset from the CSV file
#file_path = 'iri.csv'
data = pd.read_csv('/Users/abdulrehuman/Documents/Data Science/iris.csv')

# Drop rows with missing values
data = data.dropna()

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert categorical target variable to numerical using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
random_forest_model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = random_forest_model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')


# Specify the desired location to save the model
model_directory = '/Users/abdulrehuman/Documents/Data Science/'

# Save the trained model to the specific location
model_filename = 'random_forest_irismodel.pkl'
model_path = model_directory + model_filename
joblib.dump(random_forest_model, model_path)

print(f'Model saved as {model_filename}')
