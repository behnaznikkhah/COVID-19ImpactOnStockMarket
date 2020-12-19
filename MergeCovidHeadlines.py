# MergeCovidHeadlines.py

# Sai Madhuri Yerramsetti
# November 5, 2020
# Student Number: 0677671

# import required packages
import pandas as pd
import numpy as np
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt

# disable warning in case chained assignment of pandas dataframes
pd.options.mode.chained_assignment = None

# Function to combine the headline strings of monthly data, filter covid related news and merge then
def merge_and_save_data(file_dir):
    
    # initialize variables
    filepath_list = []
    news = []
    hyphen_elements = []
    hyphen_count = 0

    # Create a list of covid-19 related words
    filter_words = ['covid', 'pandemic', 'coronavirus', 'quarantine', 'cov2', 'corona virus', 'social distancing']

    # Get the list of csv files 
    file_path = os.path.abspath(file_dir)
    filepath_list = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.CSV')]

    # combines all the headlines data in a file list, take last two columns to get date and news url and name them
    unclean_data = pd.concat([pd.read_csv(f, sep="\t", header=None, engine='python') for f in filepath_list])
    corona_news = unclean_data.iloc[:, -2:]
    corona_news.columns = ['Date', 'NewsLink']

    # This loop goes through each url in NewsLink column and get the headline data
    for url in corona_news['NewsLink']:
        # split the url at '/'
        news_item = url.split('/')

        # for each element of the splitted url get the strings with more than 2 hyphens
        for item in news_item:
            if '-' in item:

                # loop to count the number of hyphens in the url strings
                for char in item:
                    if char == "-" :
                        hyphen_count += 1

                # Filter out the string with less then 2 hyphens as new websites contained one or two hyphens in them similar to headlines
                if (hyphen_count > 2):
                    hyphen_elements.append(item)
            hyphen_count = 0

        # Filter out unnecessary session ids containing hyphens present at the end of urls and get only headline data after replacing hyphen with space
        if(len(hyphen_elements) > 0):
            news.append(hyphen_elements[0].replace('-',' '))

        # If the string hyphens are none, then add news as empty string
        if(len(hyphen_elements) == 0):
            news.append('')        
        hyphen_elements = []        

    # Create a new column 'News' with headline data    
    corona_news['News'] = news
    del corona_news['NewsLink']

    # Check the news with empty string news and drop those rows
    print("Headlines with empty string: \n", corona_news[corona_news.News == ''])
    corona_news = corona_news.drop(corona_news[corona_news.News == ''].index)
    print("Dimensions of corona news dataframe is:", corona_news.shape)

    # Filter only covid-19 related headlines
    corona_news = corona_news[corona_news['News'].str.contains('|'.join(filter_words), case = False)]

    # Check the final covid-19 headlines data and check for any null values
    print("Dimensions of the filtered data", corona_news.shape)
    print(corona_news.head(10))
    print("Number of missing values: ", corona_news.isnull().values.any())

    # Save the csv file
    corona_news.to_csv(r'D:\Madhuri\Big Data Project\News data\corona_news\corona_news_jan.csv', index = False, header=True)

# If the corona news data for each months is taken seperately in merge_and_save_data(), need to run below code to combine all the months data and drop duplicate rows
def merge_corona_news_data(corona_news_path):

    file_path = os.path.abspath(corona_news_path)
    filepath_list = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]
    corona_news = pd.concat([pd.read_csv(f) for f in filepath_list])

    # dropping all duplicate values and check the dimensions and data
    corona_headlines = corona_news.drop_duplicates(subset ="News", inplace = False)
    corona_headlines = corona_headlines.reset_index(drop=True)
    print("Dimensions of final data:", corona_headlines.shape)
    print(corona_headlines.head(5))

    # Save the csv file
    corona_headlines.to_csv(r'D:\Madhuri\Big Data Project\News data\corona_news\corona_headlines.csv', index=False, header=True)
    print("Merging monthly data is finished")

if __name__ == "__main__":
    file_dir = 'D:/Madhuri/Big Data Project/News data/Jan'
    corona_news_path = 'D:/Madhuri/Big Data Project/News data/corona_news'
    merge_and_save_data(file_dir)

    # Below code should be run after executing merge_and_save_data() for all months
    # This function call merges all months data into single file after removing duplicates
    #merge_corona_news_data(corona_news_path)  
    #print("................Merging is done..................")
