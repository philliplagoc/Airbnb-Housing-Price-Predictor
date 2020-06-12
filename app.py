import numpy as np
from flask import Flask, request, jsonify, render_template
from model import predict_price # Import predict function from model.py
from itertools import chain
import pickle

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
  # Get user-inputted features 
  n_beds = int(request.form['n_beds'])
  has_beds = 1 if n_beds > 0 else 0
  n_baths = int(request.form['n_baths'])
  has_baths = 1 if n_baths > 0 else 0
  security_deposit_amount = int(request.form['security_deposit_amount'])
  has_security_deposit = 1 if security_deposit_amount > 0 else 0
  cleaning_fee_amount = int(request.form['cleaning_fee_amount'])
  has_cleaning_fee = 1 if cleaning_fee_amount > 0 else 0

  # Create the rest of the features
  # There's a specific ordering to the features according to the original data.
  features = []
  features.append(has_baths)
  features.append(has_beds)
  features.append(has_security_deposit)
  features.append(has_cleaning_fee)
  features.append(0) # availability_30 - assume none
  features.append(0) # days_as_host - assume none
  features.append(0) # extra_people - assume none
  ### Log-transform data
  features.append(0 if n_baths == 0 else np.log(n_baths))
  features.append(0 if n_beds == 0 else np.log(n_beds))
  features.append(0 if security_deposit_amount == 0 else np.log(security_deposit_amount))
  features.append(0 if cleaning_fee_amount == 0 else np.log(cleaning_fee_amount))
  features.append(0) # days_as_host - assume none
  features.append(0) # extra_people - assume none
  ### Deal with amenities. For now, assume no amenities at the listing.
  amenities = [0] * 53
  for amenity in amenities:
    features.append(amenity)
  features.append(1) # assume to be superhost
  features.append(1) # assume to have profile_pic
  ### Deal with verifications. For now, assume none.
  verifications = [0] * 15
  for verification in verifications:
    features.append(verification)
  features.append(176) # mode of host_verifications encoded 
  features.append(19) # TODO: property type
  features.append(3) # mode room type
  features.append(5) # mode bed type
  features.append(0) # mode location
  features.append(2) # mode no. accommodations

  features = np.array(features).reshape(1, len(features))

  # Make the prediction based on the features
  prediction = np.exp(predict_price(features)[0])
  
  return render_template('index.html', prediction_text = "Predicted Price per Night: ${:.2f}".format(prediction))

  
# Start the server
if __name__ == "__main__":
  app.run()

