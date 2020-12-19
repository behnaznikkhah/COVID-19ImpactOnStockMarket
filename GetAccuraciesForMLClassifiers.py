# GetAccuraciesForMLClassifiers.py

# Sai Madhuri Yerramsetti
# November 30, 2020
# Student Number: 0677671

# import required packages
import pandas as pd
import numpy as np
import os
import re
import sys
import matplotlib.pyplot as plt
from scipy.stats import randint
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# To disable warnings which arise from chained assignment
pd.options.mode.chained_assignment = None

# Classification using KNN 
def knn_classifier(X_train, X_test, y_train, y_test):
    
    # set the hyper parameters
    param_grid = {'n_neighbors' : np.arange(1, 100)}
    knn = KNeighborsClassifier()

    # Instantiate the GridSearchCV object
    knn_cv = GridSearchCV(knn, param_grid, cv=5)

    # fit the model
    knn_cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = knn_cv.predict(X_test)
  
    # Compute and print metrics
    print("Classification report of KNN is: \n", classification_report(y_test, y_pred))
    #print("Confusion Matrix of KNN is: \n", confusion_matrix(y_test, y_pred))

    # Compute and print tuned parameters and score
    print('Best parameters of KNN: {}'.format(knn_cv.best_params_))
    print('Best score of KNN: {}'.format(knn_cv.best_score_))

    # Plot the confusion matrix 
    fig = plot_confusion_matrix(knn_cv, X_test, y_test, 
                            cmap=plt.cm.Blues)
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show()
    
    return(knn_cv.best_score_)

# Classification using Decision tree classifier 
def decisiontree_classifier(X_train, X_test, y_train, y_test):
    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {"max_depth": [5, 10, 30, 50, 70, 100, None],
                  "max_features": ['auto', None],
                  "min_samples_leaf": [1, 3, 5, 7, 9, 21],
                  "criterion": ["gini", "entropy"]}

    # Instantiate a Decision Tree classifier: tree
    tree = DecisionTreeClassifier()

    # Instantiate the RandomizedSearchCV object: tree_cv
    tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

    # Fit it to the training data
    tree_cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = tree_cv.predict(X_test)
 
    # Compute and print metrics
    print("Classification report of decision tree classifier is: \n", classification_report(y_test, y_pred))
    #print("Confusion Matrix of decision tree classifier is: \n", confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    print("Best score with decision tree classifier is {}".format(tree_cv.best_score_))
    
    # Plot the confusion matrix
    fig = plot_confusion_matrix(tree_cv, X_test, y_test, 
                            cmap=plt.cm.Blues)
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show() 
    
    return(tree_cv.best_score_)

# Classification using SVM 
def svm_classifier(X_train, X_test, y_train, y_test):

    # Specify the hyperparameter space
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  
              'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'sigmoid']} 

    # Instantiate the GridSearchCV object
    svm_cv = GridSearchCV(SVC(), parameters, cv=5)

    # Fit to the training set
    svm_cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = svm_cv.predict(X_test)

    # Compute and print metrics
    print("Classification report of SVM is: \n", classification_report(y_test, y_pred))
    #print("Confusion Matrix of SVM is: \n", confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned Model Parameters of SVM: ", svm_cv.best_params_)
    print('Best score of SVM: {}'.format(svm_cv.best_score_))

    # Plot the confusion matrix 
    fig = plot_confusion_matrix(svm_cv, X_test, y_test, 
                            cmap=plt.cm.Blues)
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show()
    
    return(svm_cv.best_score_)
    
# Classification using Multinomial Naive Bayesian classifier 
def multinomialnb_classifier(X_train, X_test, y_train, y_test):
    
    # Setup the parameters and distributions to sample from: param_dist
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    # Instantiate a multinomial naive bayes classifier
    multinomialnb = MultinomialNB()

    # Instantiate the GridSearchCV object
    multinomialnb_cv = GridSearchCV(multinomialnb, param_grid, cv=5)

    # Fit it to the training data
    multinomialnb_cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = multinomialnb_cv.predict(X_test)
 
    # Compute and print metrics
    print("Classification report of multinomial naive bayes classifier is: \n", classification_report(y_test, y_pred))
    #print("Confusion Matrix of multinomial naive bayes classifier is: \n", confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned multinomial naive bayes Parameters: {}".format(multinomialnb_cv.best_params_))
    print("Best score with multinomial naive bayes classifier is {}".format(multinomialnb_cv.best_score_))
    
    # Plot the confusion matrix
    fig = plot_confusion_matrix(multinomialnb_cv, X_test, y_test, 
                            cmap=plt.cm.Blues)
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show()
    
    return(multinomialnb_cv.best_score_)

# Classification using Stochastic gradient descent classifier 
def sgd_classifier(X_train, X_test, y_train, y_test):

    # Setup the hyperparameter grid
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'penalty':['l2'], 'loss': ['modified_huber', 'hinge']}

    # Instantiate a Stochastic Gradient Descent classifier
    sgd = SGDClassifier(max_iter=500)

    # Instantiate the GridSearchCV object
    sgd_cv = GridSearchCV(sgd, param_grid, cv=5)

    # Fit the classifier to the training data
    sgd_cv.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    y_pred = sgd_cv.predict(X_test)
    
    # Compute and print metrics
    print("Classification report of Stochastic Gradient Descent classifier is: \n", classification_report(y_test, y_pred))
    #print("Confusion Matrix of Stochastic Gradient Descent classifier is: \n", confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned Model Parameters of Stochastic Gradient Descent classifier: ", sgd_cv.best_params_)
    print('Best score of Stochastic Gradient Descent classifier: {}'.format(sgd_cv.best_score_))

    # Plot the confusion matrix
    fig = plot_confusion_matrix(sgd_cv, X_test, y_test, 
                            cmap=plt.cm.Blues)
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show()
    
    return(sgd_cv.best_score_)

# Classification using Random forest classifier 
def random_forest_classifier(X_train, X_test, y_train, y_test):
    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {"max_depth": [10, 30, 50, 70, 100, None],
                "max_features": ['auto'],
                "min_samples_leaf": [1, 3, 5, 7, 9],
                "n_estimators": [50, 100, 300, 600, 800, 1000],
                "criterion": ["gini", "entropy"]}

    # Instantiate a Random forest classifier
    random = RandomForestClassifier()

    # Instantiate the RandomizedSearchCV object
    random_cv = RandomizedSearchCV(random, param_dist, cv=5)

    # Fit it to the training data
    random_cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = random_cv.predict(X_test)
 
    # Compute and print metrics
    print("Classification report of random forest classifier is: \n", classification_report(y_test, y_pred))
    #print("Confusion Matrix of random forest classifier is: \n", confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned random forest classifier Parameters: {}".format(random_cv.best_params_))
    print("Best score random forest classifier is {}".format(random_cv.best_score_))
    
    # Plot the confusion matrix
    fig = plot_confusion_matrix(random_cv, X_test, y_test, 
                            cmap=plt.cm.Blues)
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show()
    
    return(random_cv.best_score_)

# Classification using Adaboost classifier 
def adaboost_classifier(X_train, X_test, y_train, y_test):
    
    # Instantiate classifier
    adaboost = AdaBoostClassifier()

    # Specify the hyperparameter space
    parameters = {'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

    # Instantiate the GridSearchCV object
    ada_cv = GridSearchCV(adaboost, parameters, cv=5)

    # Fit to the training set
    ada_cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = ada_cv.predict(X_test)

    # Compute and print metrics
    print("Classification report of AdaBoost is: \n", classification_report(y_test, y_pred))
    #print("Confusion Matrix of AdaBoost is: \n", confusion_matrix(y_test, y_pred))

    # Print the tuned parameters and score
    print("Tuned Model Parameters of AdaBoost: ", ada_cv.best_params_)
    print("Best score of AdaBoost is {}".format(ada_cv.best_score_))    

    # Plot the confusion matrix 
    fig = plot_confusion_matrix(ada_cv, X_test, y_test, 
                            cmap=plt.cm.Blues)
    fig.ax_.set_title("Confusion Matrix plot")
    print(fig.confusion_matrix)
    plt.show()
    
    return(ada_cv.best_score_)

# Function to vectorize with 3 types of vectorizers and classify with 6 types of classifiers the data
def vectorize_and_classify(X_train, X_test, y_train, y_test):

    X_train_org = X_train
    X_test_org = X_test
    acc_list = []
    
    # Generating Bag-of-Words
    vectorizer_ng1 = CountVectorizer(ngram_range=(1,1))

    # Generating bigrams
    vectorizer_ng2 = CountVectorizer(ngram_range=(2,2))

    # Generating TF-IDF vectors
    vectorizer_tfidf = TfidfVectorizer()

    # create a list of vectorizers
    vectorizer_list = [vectorizer_ng1] # Need to change the vectorizer each time the script is run
    
    for vectorizer in vectorizer_list:
        print(".........Vectorizing the headlines using vectorizer..........")
        # fit and transform the data with the vectorizer
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Classify data using multiple classifiers
        knn_acc = knn_classifier(X_train, X_test, y_train, y_test)
        dt_acc = decisiontree_classifier(X_train, X_test, y_train, y_test)
        svm_acc = svm_classifier(X_train, X_test, y_train, y_test) # Takes relatively more time
        mnb_acc = multinomialnb_classifier(X_train, X_test, y_train, y_test)
        sgd_acc = sgd_classifier(X_train, X_test, y_train, y_test)
        rf_acc = random_forest_classifier(X_train, X_test, y_train, y_test)# Takes relatively more time
        ada_acc = adaboost_classifier(X_train, X_test, y_train, y_test) # NEED TO COMMENT THIS LINE FOR BIGRAM VECTORIZER (vectorizer_ng2) since it won't converge even after 8 hours

        # Add the accuracies in a list
        acc_list.extend([knn_acc, dt_acc, svm_acc, mnb_acc, sgd_acc, rf_acc, ada_acc]) # Need to remove ada_acc from list if running for vectorizer_ng2

        # revert back the test and training sets before next vectorization
        X_train = X_train_org
        X_test = X_test_org
    
    return(acc_list)

# Need to run this code one time for each vectorizer (total three times) to get accuracies
 if __name__ == "__main__":

    # Read the file
    corona_news = pd.read_csv('D:/Madhuri/Big Data Project/Data/final_data.csv')

    print(final_data.head(5))

    # get the unique values of countries in a list
    country_list = final_data.Country.unique().tolist()
    
    column_names = ["KNN", "DecisionTree", "SVM", "MultinomialNB", "SGD", "RandomForest", "AdaBoost"] # Need to remove AdaBoost from list if running for vectorizer_ng2

    # Create a new dataframe to save accuracies data
    acc_data = pd.DataFrame(columns = column_names)

    for country in country_list:
        print(".................Classification started..................")

        # filter the data countrywise
        model_data = final_data[final_data['Country'] == country]
        print("Country-wise data: \n", model_data.head(5))
        print("Dimensions of country-wise data: ", model_data.shape)
        model_data = model_data.reset_index(drop=True)
        print("======================== Classifying stock market labels for", country, "============================")

        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(model_data['News'], model_data['Label'], test_size=0.2, random_state=42)        

        # vectorize and classify the data
        to_append = vectorize_and_classify(X_train, X_test, y_train, y_test)

        # add the accuracy values of all machine learning algorithms as rows of acc_data dataframe created
        acc_data.loc[len(acc_data)] = to_append

    # change countries list to series and add as new column in acc_data    
    countries = pd.Series(country_list)        
    acc_data['Country'] = countries.values        
    print(acc_data.head(5))

    # save data to csv
    acc_data.to_csv(r'D:\Madhuri\Big Data Project\News data\corona_news\acc_data_BoW.csv', index = False, header=True)


