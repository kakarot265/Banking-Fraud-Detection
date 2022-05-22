from flask import Flask, render_template, url_for
import joblib
import pandas as pd
import numpy as np

from pprint import pprint

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

def preprocessing(data):
	X = data.drop('Class', axis=1)
	y = data['Class']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	robust_scaler = RobustScaler().fit(X_train)

	X_test = pd.DataFrame(robust_scaler.transform(X_test), columns=X.columns)

	return X_test, y_test

@app.route('/results')
def results():
	data = pd.read_csv('creditcard.csv')
	labels = {0: "Not Fraud", 1: "Fraud"}

	X_test, y_test = preprocessing(data)

	rf_model = joblib.load('models/random_forest_model.pkl')	
	lr_model = joblib.load('models/logistic_regression_model.pkl')	
	nb_model = joblib.load('models/naive_bayes_model.pkl')	
	
	input_tuple = X_test.sample(1) # Select a random row from X_test

	pprint(input_tuple)

	rf_prediction = labels[rf_model.predict(input_tuple)[0]]
	lr_prediction = labels[lr_model.predict(input_tuple)[0]]
	nb_prediction = labels[nb_model.predict(input_tuple)[0]]
	y_true = labels[y_test.iloc[input_tuple.index[0]]]

	new_input_tuple = input_tuple.to_dict('records')[0] # Convert DataFrame to format => [{col_name_1: value_1, col_name_2: value_2, ...}, {second_row}]

	# Converting Amount & Time values to the ones before preprocessing so that it can be displayed on webpage
	new_input_tuple['Amount'] = data.iloc[input_tuple.index[0]]['Amount']
	new_input_tuple['Time'] = data.iloc[input_tuple.index[0]]['Time']

	for col in new_input_tuple: # Round off each value to 4 decimal values so that it looks good on webpage
		new_input_tuple[col] = round(new_input_tuple[col], 4)

	return render_template("results.html", lr_prediction=lr_prediction, rf_prediction=rf_prediction, nb_prediction=nb_prediction, y_true=y_true, input_tuple=new_input_tuple)