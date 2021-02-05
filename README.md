Project Description:
In this project we attempted to analyze how the current pandemic COVID-19 impacted the stock market of fourteen countries around the world. For this, we experimented to establish a connection to stock market data with two aspects of the situation one being COVID news and other being the active number cases and deaths. We performed binary classification of stock market trends using Natural Language Processing, various machine learning algorithms and LSTM with word embedding layer to find their relationship with headlines related to COVID-19 and we tried to predict the stock price using the cases and deaths information of the pandemic. We achieved satisfactory results in binary classification with accuracies ranging from of 55% to 68% using machine learning models and in prediction of stock market price using worldwide new cases and deaths with price as features.

Dataset Used:
For this project, we collected data during the period of January 1st, 2020 to November 18th, 2020 from three sources as listed below:
1.	Daily news headlines were collected from 100% free and open GDELT events database (http://data.gdeltproject.org/events/index.html). The data contains raw information of daily events from multiple world's news media in a tab-delimited format from which ‘Date’ and ‘Url’ data is considered for this project.
2.	Stock market data was collected for fourteen different countries from ‘Investing’ website (https://ca.investing.com/indices/). This data contains ‘Date’, ‘Price’, ‘Open’, ‘High’, ‘Low’, ‘Volume’ and ‘Change %’ information of stock market index.
3.  COVID-19 data was collected from https://ourworldindata.org/coronavirus-source-data website for all the world countries and the data contains the country-wise information regarding the active cases and deaths of COVID.

Methodology:
To analyze two aspects of COVID-19 in terms of media and, active cases and deaths, we considered two approaches as mentioned below:
1.	First approach includes classification of stock market trends using the COVID-19 headline data.
2.	Second approach includes prediction of stock market price using COVID-19 active cases and deaths.

User Requirements
The user of this project will be able to check how accurately the COVID-19 headlines can classify the stock market trends of 14 countries considered for this analysis through the accuracies obtained from various classifiers and visualizations.

Functional Requirements:
When user inputs a country for which he/she wants to check how accurately different classifiers in combination with various vectorizers can classify the stock market trends, the program provides the accuracies for all combinations, therefore, user can know the best model to classify the market trends and how well it performs.

Technical Requirements:
There are no specific technical requirements for executing the code. 

Software Required:
Need Python version above 3.6 to be installed on the system with following packages:
1.	Numpy
2.	Pandas
3.	Matplotlib
4.	Scipy
5.	Scikit-learn
6.	Nltk
7.	Sklearn
8.	Keras (version 2.4.3)
9.	Tensorflow (version 2.4.0
10.	Seaborn 

Code:
Code for data collection and preparation for Classification:
•	MergeCovidHeadlines.py – This code merges daily news headline data into a single csv and includes functionality for fetching headlines from ‘url’ information, filtering COVID related headlines, dropping missing values and duplicates.
•	MergeStockPriceData.py - This code gets multiple stock market data csv files for 14 countries and merge them into single file with a new column 'Country' added.
Code for preprocessing and classification (Main Functionality):
•	CovidDataClassification.ipynb (To be executed in Google Colab) - This code classifies the stock market trends using various ML algorithms and LSTM for user entered country.
Code for visualization (Main Functionality):
•	VisualizeHeadlinesData.ipynb (To be executed in Google Colab) – This code visualizes the news headline data and classification results.
Code for obtaining accuracies for all classifiers in CSV file (Optional):
•	GetLSTMEmbeddingAccuracies.ipynb (To be executed in Google Colab) – This code gets the accuracies of LSTM with word embedding layer classification for all countries and save them in CSV file.
•	GetAccuraciesForMLClassifiers.py - This script gets the accuracies of all machine learning classifiers and save them in csv file.

Code for data collection and preparation for Prediction:
•	LSTM_part1.ipynb – This code gets multiple stock market data csv files for 13 countries and merge them into single file with a new column 'Country' added and save merged data as a new CSV file named “Market”. Also, it gets COVID-19 data, removes unnecessary columns, then saves the selected data in a new CSV file named “Covid”.
Code for price prediction (Main Functionality):
•	LSTM_Part2_newcases_exclude.ipynb (To be executed in Google Colab) - This code predicts the stock market Index price using LSTM for user entered country where only previous 30 days new number of active cases and deaths are features. 

•	LSTM_Part2_newcases_include.ipynb (To be executed in Google Colab) - This code predicts the stock market Index price using LSTM for user entered country where not only previous 30 days new number of active cases and deaths are features but also prior Index price values was used as features.

•	LSTM_Part2_totalcases_exclude.ipynb (To be executed in Google Colab) - This code predicts the stock market Index price using LSTM for user entered country where only previous 30 days total number of active cases and deaths are features.

•	LSTM_Part2_totalcases_include.ipynb (To be executed in Google Colab) - This code predicts the stock market Index price using LSTM for user entered country where not only previous 30 days total number of active cases and deaths are features but also prior Index price values was used as features.
Code for visualization (Main Functionality):
•	Part2_Visualization.ipynb (To be executed in Google Colab) – This code visualizes the stock market and COVID-19 data.

