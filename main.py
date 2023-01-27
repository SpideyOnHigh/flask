from flask import Flask ,request ,jsonify
import pickle
import numpy as np
from sklearn import *
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
enc = pickle.load(open('enc.pkl','rb'))

@app.route('/recommend',methods = ['POST'])
def recommend():
    veg_or_nonveg = request.form.get('veg_or_nonveg')
    taste = request.form.get('taste')
    prep_time = request.form.get('prep_time')
    budget = request.form.get('budget')
    type = request.form.get('type')

    input_features = pd.DataFrame(
        {'veg_or_nonveg': [veg_or_nonveg], 'Taste': [taste], 'Prep Time': [prep_time], 'Budget': [budget],
         'Type': [type]})


    # One-hot encode the input features
    input_features = enc.transform(input_features).toarray()

    result = model.predict(input_features)
    return jsonify({'result': result.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
