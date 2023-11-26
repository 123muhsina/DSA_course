from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
#dataset_path = 'modeldata/iris.csv'
df = pd.read_csv("modeldata/iris.csv")

# Drop rows with missing values
df = df.dropna()

# Split the data into features (X) and target (y)
X = df[['SL', 'SW', 'PL', 'PW']]
y = df['Classification']

# Train a simple RandomForestClassifier (replace this with your actual model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Extract feature values from the form
    SL = float(request.form['SL'])
    SW = float(request.form['SW'])
    PL = float(request.form['PL'])
    PW = float(request.form['PW'])

    # Perform classification using the pre-trained model
    predicted_class = model.predict([[SL, SW, PL, PW]])[0]

    return render_template('index.html', result=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
