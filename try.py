import streamlit as st
from flask import Flask
import pandas as pd
import pickle

# Initialize your Flask app as usual
app = Flask(__name__)


@st.cache(show_spinner=False)
def load_models():
    enc = pickle.load(open('enc.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return enc, model


enc, model = load_models()


def predict(veg_or_nonveg, taste, prep_time, budget, type):

    if(prep_time == '10 Minutes'):
        prep_time = '10Mins'
    elif(prep_time == '15-20 Minutes'):
        prep_time = '1520 Mins'
    elif(prep_time == '30 Minutes'):
        prep_time = '30 mins'
    elif(prep_time == '1 hour'):
        prep_time = '1 hour'


    veg_or_nonveg = str(veg_or_nonveg)
    taste = str(taste)
    prep_time = str(prep_time)
    budget = str(budget)
    type = str(type)
    input_features = pd.DataFrame(
        {'veg_or_nonveg': [veg_or_nonveg], 'Taste': [taste], 'Prep Time': [prep_time], 'Budget': [budget],
         'Type': [type]})

    print(input_features)
    input_features = enc.transform(input_features).toarray()
    result = model.predict(input_features)
    return result


def main():
    st.title("Food Recommendation App")

    veg_or_nonveg = st.radio("Select food type:", ('Vegetarian', 'nonveg', 'eggbased'))
    taste = st.selectbox("Select taste preference:", ('Salty', 'Umami', 'Sweet', 'Spicy'))
    prep_time = st.selectbox("Enter maximum preparation time (in minutes):",('10 Minutes','15-20 Minutes','30 Minutes','1 hour'))
    budget = st.selectbox("Select budget:", ('Low budget', 'Avg budget', 'Money does not matter to me'))
    type = st.selectbox("Select food type:", ('Snacks', 'Light Meal', 'Heavy Meal'))

    if st.button("Recommend"):
        result = predict(veg_or_nonveg, taste, prep_time, budget, type)
        st.success(f"Recommended dish: {result}")
    st.markdown("""<h1 style="font-size: 50px; color: #BBFF00">Created by <a style= "text-decoration: none;" href ="https://github.com/SpideyOnHigh">Kalp Shah</a></h1>""",unsafe_allow_html=True)


if __name__ == '__main__':
    main()
