from flask import Flask, render_template, request, jsonify
import pickle  # For loading the trained model

app = Flask(__name__)

# Load the pre-trained model (replace with your actual model)
with open('model/random_forest_irismodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Extract feature values from the JSON request
    data = request.get_json()
    feature_values = [float(data[f]) for f in ['SL', 'SW', 'PL', 'PW']]

    # Perform classification using the pre-trained model
    predicted_class = model.predict([feature_values])[0]

    return jsonify(predicted_class)

if __name__ == '__main__':
    app.run(debug=True)