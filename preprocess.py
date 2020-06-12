#!/usr/bin/env python
# coding: utf-8


# Import packages
import pandas as pd
import numpy as np

from datetime import datetime, date
from scipy import stats



# Get data
df = pd.read_csv('data/listings.csv')



# Preprocess data

### Step 1) Delete variables not inherently helpful ####
delete_vars = [col for col in df.columns if len(df[col].unique()) == 1]
df_clean = df.drop(delete_vars, axis = 1)
# Delete URLs
delete_vars = [col for col in df_clean.columns if 'url' in col]
df_clean = df_clean.drop(delete_vars, axis = 1)

# Delete IDs 
delete_vars = [col for col in df_clean.columns if 'id' in col]
df_clean = df_clean.drop(delete_vars, axis = 1)

# Delete full textual data 
full_text_columns = ['description', 'host_name', 'name', 'summary', 'space', 'neighborhood_overview', 'notes', 'house_rules', 'access', 'interaction', 'transit', 'host_about']
df_clean = df_clean.drop(full_text_columns, axis = 1)

# Delete scraping information
df_clean = df_clean.drop(['last_scraped'], axis = 1)
df_clean = df_clean.drop(['calendar_last_scraped'], axis = 1)



### Step 2) Missing variables ###
df_clean = df_clean.drop(['license'], axis = 1)
df_clean = df_clean.drop(['square_feet'], axis = 1)
df_clean = df_clean.drop(['monthly_price', 'weekly_price'], axis = 1)
df_clean['host_response_time'] = df_clean['host_response_time'].fillna('no response')
df_clean['host_response_rate'] = df_clean['host_response_rate'].fillna('no response')

"""
Convert dollars to a float value. Used in applying to a column.

Parameters:
dollar_str - The dollar string to convert.
"""
def dollar_str_to_int(dollar_str):
    return float(dollar_str.replace('$', '').replace(',', ''))

df_clean['security_deposit'] = df_clean['security_deposit'].fillna('$0')
df_clean['security_deposit'] = df_clean['security_deposit'].apply(dollar_str_to_int)

df_clean = df_clean.drop(['host_neighbourhood'], axis = 1)
df_clean['cleaning_fee'] = df_clean['cleaning_fee'].fillna('$0')
df_clean['cleaning_fee'] = df_clean['cleaning_fee'].apply(dollar_str_to_int)

df_clean['review_scores_checkin'] = df_clean['review_scores_checkin'].fillna('no review')
df_clean['review_scores_value'] = df_clean['review_scores_value'].fillna('no review')
df_clean['review_scores_location'] = df_clean['review_scores_location'].fillna('no review')
df_clean['review_scores_communication'] = df_clean['review_scores_communication'].fillna('no review')
df_clean['review_scores_accuracy'] = df_clean['review_scores_accuracy'].fillna('no review')
df_clean['review_scores_cleanliness'] = df_clean['review_scores_cleanliness'].fillna('no review')
df_clean['review_scores_rating'] = df_clean['review_scores_rating'].fillna('no review')

df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

df_clean['first_review'] = pd.to_datetime(df_clean['first_review']) # Month first 
df_clean['last_review'] = pd.to_datetime(df_clean['last_review']) 

df_clean['days_since_first_review'] = (datetime(2019, 8, 8) - df_clean['first_review']).dt.days
df_clean['days_since_last_review'] = (datetime(2019, 8, 8) - df_clean['last_review']).dt.days

df_clean['days_since_first_review'] = df_clean['days_since_first_review'].fillna(0)
df_clean['days_since_last_review'] = df_clean['days_since_last_review'].fillna(0)

df_clean = df_clean.drop(['neighbourhood', 'zipcode', 'state', 'city'], axis = 1)

df_clean = df_clean.drop(['host_location', 'jurisdiction_names'], axis = 1)
df_clean['market'] = df_clean['market'].fillna(df_clean['market'].describe()['top'])

df_clean['host_has_profile_pic'] = df_clean['host_has_profile_pic'].fillna(df_clean['host_has_profile_pic'].describe()['top'])

df_clean = df_clean.drop(['host_total_listings_count', 'host_listings_count'], axis = 1)

df_clean['host_is_superhost'] = df_clean['host_is_superhost'].fillna(df_clean['host_is_superhost'].describe()['top'])

df_clean['host_since'] = pd.to_datetime(df_clean['host_since']) 
df_clean['days_as_host'] = (datetime(2019, 8, 8) - df_clean['host_since']).dt.days
df_clean = df_clean.dropna(subset = ['host_since'])

df_clean = df_clean.dropna(subset = ['bedrooms', 'beds', 'bathrooms', 'cancellation_policy'])



### Step 3) Preprocessing Y
df_clean['price'] = df['price'].apply(dollar_str_to_int)
df_clean = df_clean[df_clean.price != 0]
df_clean['log_price'] = np.log(df_clean['price'])



### Step 4) Dealing with Datetime variables
df_clean = df_clean.drop(['host_since', 'last_review', 'first_review'], axis = 1)



### Step 5) Location Variables
location_vars = ['street', 'neighbourhood_cleansed', 'market', 'smart_location', 'latitude', 'longitude', 'is_location_exact']



### Step 6) Categorical Variables
categ_vars = [col for col in df_clean.columns if df_clean[col].dtype == 'object' and col not in location_vars]

"""
Categorizes the given score by using the above scheme.
Applies only to review_scores_rating.

Parameters:
score - The string to preprocess.
"""
def categorize_score(score):
    if (score == 'no review') | (score == 'no response'):
        return 'Missing'
    
    if int(score) >= 95:
        return 'A+'
    elif int(score) >= 90 and int(score) <= 94:
        return 'A'
    elif int(score) >= 80 and int(score) <= 89:
        return 'B'
    elif int(score) >= 70 and int(score) <= 79:
        return 'C'
    elif int(score) >= 60 and int(score) <= 69:
        return 'D'
    elif int(score) <= 59:
        return 'F'

"""
Alteration of the above function for use with a different range of ratings.

Parameters:
score - The string to preprocess.
"""
def categorize_score_alt(score):
    if score == 'no review':
        return 'Missing'
    elif int(score) == 10:
        return 'A+'
    elif int(score) == 9:
        return 'A'
    elif int(score) == 8:
        return 'B'
    elif int(score) == 7:
        return 'C'
    elif int(score) == 6:
        return 'D'
    elif int(score) <= 5:
        return 'F'
    
for review_var in [var for var in df_clean.columns if 'review_scores_' in var]:
    if review_var == 'review_scores_rating':
        df_clean['cleaned_' + review_var] = df_clean[review_var].apply(categorize_score)
    else:
        df_clean['cleaned_' + review_var] = df_clean[review_var].apply(categorize_score_alt)
        
# Keep track of these new variables
categ_vars.append('cleaned_review_scores_rating')
categ_vars.append('cleaned_review_scores_accuracy')
categ_vars.append('cleaned_review_scores_cleanliness')
categ_vars.append('cleaned_review_scores_checkin')
categ_vars.append('cleaned_review_scores_communication')
categ_vars.append('cleaned_review_scores_location')
categ_vars.append('cleaned_review_scores_value')

# Array to keep track of boolean variables
categ_bool_vars = []

# Boolean encode
df_clean['host_is_superhost'] = df_clean['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)
categ_bool_vars.append('host_is_superhost')

uniq_amenities = []

for row in df_clean['amenities']:
    for amenity in row[1:-1].replace('"', '').split(','):
        if amenity not in uniq_amenities:
            uniq_amenities.append(amenity)

# Delete those with translation missing
uniq_amenities = [amenity for amenity in uniq_amenities if 'translation missing' not in amenity]
# Delete empty strings
uniq_amenities = [amenity for amenity in uniq_amenities if amenity]

# Combine the same amenities into one eg. internet and Wifi are the same thing
df_clean.loc[df_clean['amenities'].str.lower().str.contains('bbq'), 'bbq'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('outlet covers'), 'outlet_covers'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('gym'), 'gym'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('mobile hoist'), 'mobile_hoist'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('air purifier'), 'air_purifier'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('doorman'), 'doorman'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('building staff'), 'building_staff'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('hot water kettle'),'hot_water_kettle'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('suitable for events'),'event_suitable'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('air conditioning'),'ac'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('other'),'other'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('room-darkening'),'room_shades'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('luggage dropoff'),'luggage_dropoff'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('charger'),'ev_charger'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('pocket wifi|wifi|internet|ethernet'),'internet'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('wireless intercom|buzzer'),'buzzer'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('heating'),'heating'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('washer|dryer|washer / dryer'),'washer_or_dryer'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('shower|essentials|shampoo|bed|pillows|toilet|mattress|shower|bathroom'),'bed_and_bath_essentials'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('keypad|lock on bedroom door|private|lockbox'),'privacy'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('24-hour'),'24_hr_checkin'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('hangers'),'hangers'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('hair dryer'),'hair_dryer'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('iron'),'iron'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('laptop friendly'),'laptop_friendly'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('hot water'),'hot_water'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('garden|backyard|lake|beach|waterfront|beachfront|ski-in/ski-out'),'outdoor'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('wheelchair|accessible|no stairs|well-lit|flat path|wide|extra space|access|disabled'),'accessible'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('single level home'),'single_level_home'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('handheld shower head'),'handheld_shower_head'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('parking'),'parking'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('tv|cable tv'),'television'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('refrigerator|kitchen|microwave|cooking basics|dishes and silverware|dinnerware'),'kitchen_basics'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('long term stays allowed'),'long_term_stays'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('elevator'),'elevator'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('indoor fireplace'),'indoor_fireplace'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('dryer'),'dryer'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('smoking allowed'),'smoking_allowed'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('coffee maker'),'coffee_maker'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('dishwasher'),'dishwasher'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('oven'),'oven'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('stove'),'stove'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('bathtub'),'bathtub'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('cleaning before checkout'),'cleaning_before_checkout'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('breakfast'),'breakfast'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('self check-in'),'self_checkin'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('tv|cable tv'),'television'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('other pet(s)|Dog(s)|Cat(s)|Pets live on this property|pet'), 'pet_accessibility'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('gates|high chair|crib|baby monitor|child|children|kids|kids|baby|changing table|guards|gates'),'kid_friendly'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('game console'),'high_end_electronics'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('host greets you'),'host_greets_you'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('carbon monoxide alarm|fire extinguisher|first aid kit|smoke alarm|safety card'),'safety_features'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('patio or balcony'),'patio_or_balcony'] = 1
df_clean.loc[df_clean['amenities'].str.lower().str.contains('hot tub|pool'),'hot_tub_or_pool'] = 1

# Replace NaNs with 0s in the new columns
df_clean.loc[:, 'bbq':] = df_clean.loc[:, 'bbq':].fillna(0)

amenity_bool_vars = [col for col in df_clean.loc[:, 'bbq':].columns]

# Boolean encode
df_clean['host_has_profile_pic'] = df_clean['host_has_profile_pic'].apply(lambda x: 1 if x == 't' else 0)
categ_bool_vars.append('host_has_profile_pic')

# Clean response_rate
df_clean['cleaned_host_response_rate'] = df_clean['host_response_rate'].str.replace('%', '').apply(categorize_score)
categ_vars.append('cleaned_host_response_rate')

# More boolean variables
df_clean['is_location_exact'] = df_clean['is_location_exact'].apply(lambda x: 1 if x == 't' else 0)
categ_bool_vars.append('is_location_exact')

df_clean['instant_bookable'] = df_clean['instant_bookable'].apply(lambda x: 1 if x == 't' else 0)
categ_bool_vars.append('instant_bookable')

df_clean['require_guest_phone_verification'] = df_clean['require_guest_phone_verification'].apply(lambda x: 1 if x == 't' else 0)
categ_bool_vars.append('require_guest_phone_verification')

df_clean['require_guest_profile_picture'] = df_clean['require_guest_profile_picture'].apply(lambda x: 1 if x == 't' else 0)
categ_bool_vars.append('require_guest_profile_picture')

uniq_verifications = []

for row in df_clean['host_verifications']:
    for verification in row[1:-1].replace('"', '').replace(' ', '').split(','):
        if verification not in uniq_verifications:
            uniq_verifications.append(verification)
            
# Delete empty strings
uniq_verifications = [v for v in uniq_verifications if v]

# One hot encode each verification
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('email|work_email'), 'email'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('phone'), 'phone'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('reviews'), 'jumio'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('government'), 'government'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('selfie'), 'selfie'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('identity_manual'), 'id_manual'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('facebook'), 'fb'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('google'), 'google'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('manual'), 'manual'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('sent_id'), 'sent_id'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('kba'), 'kba'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('weibo'), 'weibo'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('selfie'), 'selfie'] = 1
df_clean.loc[df_clean['host_verifications'].str.lower().str.contains('sesame'), 'sesame'] = 1

# Get rid of NaNs
df_clean.loc[:, 'email':] = df_clean.loc[:, 'email':].fillna(0)

# Keep track of these
verification_bool_vars = [col for col in df_clean.loc[:, 'email':].columns]

# Remove unneeded variables
categ_vars.remove('calendar_updated')
categ_vars.remove('review_scores_rating')
categ_vars.remove('review_scores_accuracy')
categ_vars.remove('review_scores_cleanliness')
categ_vars.remove('review_scores_checkin')
categ_vars.remove('review_scores_communication')
categ_vars.remove('review_scores_location')
categ_vars.remove('review_scores_value')
categ_vars.remove('host_response_rate')
categ_vars.remove('extra_people')
categ_vars.remove('amenities')

# Add location variables
categ_vars.append('neighbourhood_cleansed')
categ_vars.append('market')

# Exclude one hot variables
categ_vars = [var for var in categ_vars if var not in categ_bool_vars]

"""
Encodes the categorical variables using ordinal encoding based on the magnitude of the median price.
"""
def encode(frame, feature):
    # Get the median price for each level and sort
    meds = frame[[feature, 'price']].groupby(feature).median()['price']
    meds = meds.sort_values() # Sort in ascending order
    # Create a mapping as a dictionary where keys are labels and values are integers
    mapping = range(1, len(meds) + 1)
    enc_dict = {}
    i = 0
    for f in meds.index:
        enc_dict[f] = mapping[i]
        i = i + 1
    
    for category, mapping in enc_dict.items():
        # Create a new column using the integer mapping sorted by median price
        frame.loc[frame[feature] == category, feature + '_enc'] = mapping
        
categ_vars_enc = []
for var in categ_vars:  
    encode(df_clean, var)
    categ_vars_enc.append(var + '_enc')
    
# This variable was not significant in determining price
categ_vars_enc.remove('market_enc')



### Step 7) Qualitative Variables
"""
Returns True if the data given is normal ie. the p-value is less than 0.05. 
If not, then return False.

Parameters:
data - The data to check the normality of.
"""
def is_normal(data):
    _, p = stats.normaltest(data)
    if p < 0.05:
        return False
    else:
        return True

df_clean = df_clean.drop(['minimum_minimum_nights', 'minimum_maximum_nights', 'maximum_minimum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews_ltm', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms'], axis = 1)

# What are the qualitative variables?
quant_vars = [col for col in df_clean.columns if df_clean[col].dtype != 'object' and col not in location_vars and col not in categ_vars]
quant_vars.remove('log_price')

# Exclude from amenity_bool_var, categ_bool_vars, and verification_bool_vars
quant_vars = [var for var in quant_vars if var not in amenity_bool_vars]
quant_vars = [var for var in quant_vars if var not in categ_bool_vars]
quant_vars = [var for var in quant_vars if var not in verification_bool_vars]

df_clean['extra_people'] = df_clean['extra_people'].fillna('$0')
df_clean['extra_people'] = df_clean['extra_people'].apply(dollar_str_to_int)

quant_vars.append('extra_people')

# Tidy up quant_vars
quant_vars = [var for var in quant_vars if var not in amenity_bool_vars]
quant_vars = [var for var in quant_vars if var not in categ_bool_vars]
quant_vars = [var for var in quant_vars if var not in verification_bool_vars]
quant_vars = [var for var in quant_vars if var not in categ_vars_enc]
quant_vars.remove('market_enc')

# Holds the names of the columns that have a 0
quant_bool_vars = []

# Holds the names of the columns that have been encoded (transformed or not)
quant_vars_enc = []

for var in [bool_var for bool_var in quant_vars if 0 in df_clean[bool_var].values]:
    # Boolean encode
    df_clean['has_' + var] = df_clean[var].apply(lambda x: 1 if x > 0 else 0)
    quant_bool_vars.append('has_' + var)
    
    # Log transform
    if is_normal(df_clean[var]):
        # No need for any transforming
        quant_vars_enc.append(var)
    else:
        # Log transform, create a new column, and append the name to the list
        new_col = var + '_enc'
        df_clean[new_col] = df_clean[var].apply(lambda x: np.log(abs(x)) if x > 0 else 0)
        quant_vars_enc.append(new_col)
        
df_clean['accommodates_enc'] = df_clean[var].apply(lambda x: np.log(abs(x)) if x > 0 else 0)
quant_vars_enc.append('accommodates_enc')



### Step 8) Ordinally encode accommodates
training_vars = quant_bool_vars + quant_vars_enc + amenity_bool_vars + categ_bool_vars + verification_bool_vars + categ_vars_enc
model_df = df_clean[training_vars + ['log_price']]

# Clean up training_df
model_df.drop(['bedrooms_enc', 'has_bedrooms', 'accommodates_enc'], axis = 1, inplace = True)

# Ordinally encode accommodates
df_clean['accommodates'] = df_clean['accommodates'].apply(str)
encode(df_clean, 'accommodates')
categ_vars_enc.append('accommodates_enc')

model_df['accommodates_enc'] = df_clean['accommodates_enc']



### Step 9) Remove features unknown to host prior to listing 
vars_to_remove = ['has_number_of_reviews', 'has_reviews_per_month',
                  'has_days_since_first_review', 'has_days_since_last_review', 
                  'availability_30_enc', 'number_of_reviews_enc', 'reviews_per_month_enc', 'days_since_first_review_enc',
                 'days_since_last_review_enc', 'is_location_exact', 'host_response_time_enc', 'instant_bookable', 
                 'cancellation_policy_enc', 'cleaned_review_scores_rating_enc', 'cleaned_review_scores_accuracy_enc',
                 'cleaned_review_scores_cleanliness_enc', 'cleaned_review_scores_checkin_enc', 
                  'cleaned_review_scores_communication_enc', 'cleaned_review_scores_location_enc', 
                  'cleaned_review_scores_value_enc', 'cleaned_host_response_rate_enc']

final_df = model_df.drop(vars_to_remove, axis = 1)



# Export the cleaned data
final_df.to_csv("cleaned_listings.csv")
