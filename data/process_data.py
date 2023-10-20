import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''load data from input filepaths and return a pandas dataframe'''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id', how='outer')
    return df


def clean_data(df):
    '''process necessary steps to clean data, return pandas dataframe'''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # use first row to extract a list of new column names for categories.
    row = categories.iloc[0]
    category_colnames = row.str.split('-', expand=True).iloc[:,0]
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    # replace 'categories' column
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    # remove duplicates
    df.drop_duplicates(inplace=True)
    # remove 2 values if any
    df = df[df['related'] != 2]
    return df


def save_data(df, database_filename):
    '''save @df as @database_filename'''
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('data', engine, index=False, if_exists='replace')


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
