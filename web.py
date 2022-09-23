from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction= model.predict(final_features)
    
    
    return render_template('Result.html', prediction_text='Predicted Class: {}'.format(prediction))

if __name__ == "__main__":
   app.run(debug=True)