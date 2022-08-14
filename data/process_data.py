import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads two csv files with pandas 'pd.read_csv' method, combines them and returns the combined dataframe.

    Args:
        messages_filepath (filepath): filepath for the messages csv file
        categories_filepath (filepath): filepath for the categories csv file

    Returns:
        DataFrame: pandas dataframe of the combined datasets 
    """

    # read datasets
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # merging datasets on id column
    df = pd.merge(df_messages, df_categories, on='id')

    return df 


def clean_data(df):
    """Receives a pandas dataframe and apply a series of transformations in it. Returns the cleaned dataframe. 

    Args:
        df (DataFrame): Pandas DataFrame

    Returns:
        DataFrame: Pandas dataframe 
    """

    # create categories dataframe 
    categories = df.categories.str.split(';', expand=True)

    # get the first row of the dataset
    row = categories.loc[0]

    # get the columns names from the first row
    category_names = row.apply(lambda x: x.split('-')[0])

    # change the columns names in the catagories dataset
    categories.columns = category_names

    # convert category values to binary encoding
    for col in categories.columns:

        # gets the last digit from each value
        categories[col] = categories[col].apply(lambda x: str(x).split('-')[1])

        # change the column datatype to int 
        categories[col] = categories[col].astype('int64')

    # drop categories column from dataframe
    df.drop('categories', axis=1, inplace=True)

    # concats categories and df 
    df = pd.concat([df, categories], axis=1)

    # drop duplicates from df
    df.drop_duplicates(inplace=True)

    return df 

    


def save_data(df, database_filename):
    """Stores a pandas dataframe into a sqlite database.

    Args:
        df (DataFrame): Pandas dataframe that will be stored in the sqlite database
        database_filename (str): sqlite database name
    """

    # creates engine
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # saves dataset to the database
    df.to_sql('messages', engine, index=False)



def main():
    """
    Runs the entire ETL process by applying the 'load_data', 'clean_data' and 'save_data' functions
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()