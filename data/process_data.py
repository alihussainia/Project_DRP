import sys
import pandas as pd
import csv
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loading Data
    Load Messages and categories file 
    and return a merged dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    return pd.merge(messages, categories, on="id")

def clean_data(df):
    """
    Split into individual category columns and create
    a new column with the identifier given in newly created column and clean the dataset
    """
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: str(x)[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: str(x)[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    categories = categories.reset_index(drop=True)
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1).reset_index(drop=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    #Drop duplicates
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    return df

def save_data(df, database_filename):
    """save_date
    Save dataframe df to sqlite database with a database_filename
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
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