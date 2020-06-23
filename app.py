import numpy as np
from flask import Flask, request, jsonify, render_template
from model import predict_price # Import predict function from model.py
from itertools import chain
import pickle

app = Flask(__name__)

# Dict for accommodations ordinal mapping
acc_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '8': 6, '6': 7, '7': 8, '9': 9, '12': 10, '17': 11, '10': 12, '16': 13, '14': 14, '11': 15, '30': 16, '18': 17}

@app.route('/')
def home():
  return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
  # Get user-inputted features 
  n_accommodates = acc_dict[str(request.form['n_accommodates'])]
  n_beds = int(request.form['n_beds'])
  has_beds = 1 if n_beds > 0 else 0
  n_baths = int(request.form['n_baths'])
  has_baths = 1 if n_baths > 0 else 0
  security_deposit_amount = int(request.form['security_deposit_amount'])
  has_security_deposit = 1 if security_deposit_amount > 0 else 0
  cleaning_fee_amount = int(request.form['cleaning_fee_amount'])
  has_cleaning_fee = 1 if cleaning_fee_amount > 0 else 0
  property_type = request.form['property_type']
  extra_people = int(request.form['extra_people_amount'])
  has_extra_people = 1 if extra_people > 0 else 0
  room_type = request.form['room_type']

  # Create the rest of the features
  # There's a specific ordering to the features according to the original data.
  features = []
  features.append(has_baths)
  features.append(has_beds)
  features.append(has_security_deposit)
  features.append(has_cleaning_fee)
  features.append(0) # availability_30 - assume none
  features.append(0) # days_as_host - assume none
  features.append(has_extra_people) # has_extra_people - assume none
  ### Log-transform data
  features.append(0 if n_baths == 0 else np.log(n_baths))
  features.append(0 if n_beds == 0 else np.log(n_beds))
  features.append(0 if security_deposit_amount == 0 else np.log(security_deposit_amount))
  features.append(0 if cleaning_fee_amount == 0 else np.log(cleaning_fee_amount))
  features.append(0) # days_as_host - assume none
  features.append(extra_people) # extra_people - assume none
  ### Assume no amenities
  amenities = [0] * 53
  for amenity in amenities:
    features.append(amenity)
  features.append(0) # assume to not be a superhost
  features.append(1) # assume to have profile_pic
  ### Assume no verifications
  verifications = [0] * 15
  for verification in verifications:
    features.append(verification)
  features.append(176) # mode of host_verifications encoded 
  features.append(property_type) 
  features.append(room_type) 
  features.append(5) # mode bed type
  features.append(0) # mode location
  features.append(n_accommodates) 

  features = np.array(features).reshape(1, len(features))

  # Make the prediction based on the features
  prediction = np.exp(predict_price(features)[0])
  
  return render_template('index.html', prediction_text = "Predicted Price per Night: ${:.2f}".format(prediction))

  
# Start the server
if __name__ == "__main__":
  app.run()

