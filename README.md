# Disaster Response Pipelines
### Udacity's Data Scientist Nanodegree Project
-------------------------

## Table of Contents 
1. [Project Overview](#overview)
2. [The Problem that Needed to be Solved](#problem)
3. [File Descriptions](#files)
4. [Used Frameworks](#frameworks)
5. [Acknowledgements](#acknowledgements)
6. [Instructions](#instructions)

## Project Overview <a name="overview"><a/>
**The project is composed of analyzing Figure Eight's dataset about messages sent during natural disaster and classify them into the correct disaster category. The project is divided into 3 parts:**
1. Creating an ETL Pipeline that reads the data, clean it with nlp techniques and saves it into a database
2. Creating an Machine Learning Pipeline that trains a supervised classification model in order to correctly predict the message's class
3. Runing an Flask app that contains a few visualizations of the dataset as well as a text box to add new messages and get the mode's classification on it

## The Problem that Needed to be Solved <a name="problem"><a/>
**When a natural disaster happens, the disaster response professionals need to act quickly in order to save as much lives as they can. In order to do there jobs correctly, they need to read milions of messages they received, either direct or from social media. The problem is that it's not easy to filter for disaster-related messages through humam scan or simple techniques like keyword search. So point of the project is to build a supervised machine learning model that can automatically classify the messages into the correct disaster category, if there is one.**

## File Descriptions <a name="files"></a>
1. App Folder
    - Templates: Contains the HTML files used for building the website 
    - run.py: File used to build the Flask app and build the visualizations
2. Data
    - disaster_categories.csv and disaster_messages.csv: Csv files that contains the data used to train and validate the model
    - process_data.py: ETL pipeline file 
    - DisasterResponse.db: Data Base where the cleaned data was saved after the ETL process
3. Models
    - train_classifier.py: Machine Learning pipeline created to build, train and test the model
    - classifier.pkl: File with the saved model after the training
4. Development-Notebooks
    - ETL_pipeline.ipynb: Notebook used to develop the ETL pipeline 
    - ML_pipeline.ipynb: Notebook used to develop the Machine Learning pipeline

## Used Framewors <a name="frameworks"></a>
- Pandas
- NumPy
- Nltk
- SKlearn
- Matplotlib
- Seaborn
- Plotly
- SQLAlchemy
- Re
- Json
- Pickle
- Sys
- Joblib

## Acknowledgements <a name="acknowledgements"></a>
- Multiclass Classification: https://scikit-learn.org/stable/modules/multiclass.html
- Adaboost Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
- Hyperparameter tuning: https://www.oreilly.com/library/view/machine-learning-with/9781789343700/0d1439c3-f90b-4cb3-9e76-0f3cc0dcd6f9.xhtml

## Instructions <a name="instructions"></a>
- 1. Run the ETL Pipeline: `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- 2. Run the Machine Learning Pipeline: `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- 3. Run the app: `python3 run.py`
