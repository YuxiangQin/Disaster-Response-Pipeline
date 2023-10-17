# Disaster Response Pipeline Project

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Follow instruction in terminal to open the homepage, usually http://127.0.0.1:3000 would take you to it.

## File descriptions
### data
This folder contains:
- 2 csv data files, raw messages and categories data used to train model and plot visualizations.
- `process_data.py`, script for loading and cleaning data, then save in .db file.
- `DisasterResponse.db`, file created from `process_data.py` script.  
### models
This folder contains:
- `train_classifier.py`, scripts to train, evaluate and save model.
- `classifier.pkl`, this file will be created after running `train_classifier.py`, stores trained model.
### app
This folder contains:
- templates folder, contains necessary html templates used on homepage.
- `run.py`, script to generate homepage, visualizations are created here. 
