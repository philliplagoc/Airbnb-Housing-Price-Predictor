# Airbnb Price per Night Predictor
This project helps new Airbnb hosts decide a price for their listing. Users input several features about their listing and my model will predict a price they can potentially use. This is an extension of my Amsterdam Airbnb data analysis project, and is still a work in progress.

[Try it out yourself](https://airbnb-price-predictor.herokuapp.com/)!

In this project, I used public Airbnb data from [here](http://insideairbnb.com/get-the-data.html). My model employed ensemble techniques. I used 4 boosting algorithms: XGBoost, LightGBM, AdaBoost, and GradientBoost. Then, I bagged them together by averaging their predictions.

My final model achieved an MAE of 0.25, and deployed it using Flask.

## Getting Started
If you want to run the process all 
1. Clone this repo.
2. Install the requirements: `pip install -r requirements.txt`
3. Preprocess the data: `python preprocess.py`
4. Model the data: `python model.py`
5. Run the website: `python app.py`
6. Go to `http://127.0.0.1:5000/` to see the webpage.

## Featured Notebook
* [Airbnb Data Analysis](https://github.com/philliplagoc/AirBnb_Data_Analysis)

This notebook explains the reasons behind the preprocessing and modelling I used in `preprocess.py` and `model.py`, respectively. 

