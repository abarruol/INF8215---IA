"""
Team:
<<<<< Nitsugualuo >>>>>
Authors:
<<<<<  Barruol Augustin - 2161214 >>>>>
<<<<< DE LA GRANDIERE Bathylle - MATRICULE #2 >>>>>
"""

# cd Desktop/Cours/INF8215\ -\ IA/tp3
# python3 main.py --train_file data/train.csv --test_file data/test_public.csv --prediction_file data/sample_submission.csv

# Code implémenté par :
    # Augustin BARRUOL - 2161214
    # Bathylle DE LA GRANDIERE - 2166208

from wine_testers import WineTester

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



class MyWineTester(WineTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.rf = RandomForestClassifier()
        self.rf_optimized = RandomForestClassifier()

    
    def X_data_preprocessing(self, X_data):
        X_processed = X_data.copy()
        
        # Changer 'color' == 'red' en '1' et 'color' == 'white' en '0'
        N = len(X_data)
        for i in range (N):
            if X_data[i][1] == 'red': X_data[i][1] = '1'
            if X_data[i][1] == 'white': X_data[i][1] = '0'

        return X_processed
    
    
    def plot_learning_curve(self, estimator, X, Y):
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plt.figure()
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            Y,
            cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            return_times=True,
        )  
        
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        plt.legend(loc="best")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Score en fonction du nombre de données")
        plt.show()

        
    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        # TODO: entrainer un modèle sur X_train & y_train
        
        # Convert 'white' and 'red' into binary values
        X_train_processed = self.X_data_preprocessing(X_train)
        
        # Remove the id's of the train data 
        X_train_processed = [elem[1:] for elem in X_train]
        Y_train_processed = [elem[1] for elem in y_train]
                
        # Initialize objects
        std_slc = StandardScaler()

        # Standardize the data
        std_slc.fit_transform(X_train_processed)
        
        # Initialize the model
        #rf = self.rf
        
        # Optimization of the hyperparameters of the model
        """n_components = [3, 6 , 9, 12]
        criterion = ['gini, 'entropy']
        max_depth = [None, 5, 10, 15, 25]
        n_estimators = [10, 50, 250, 500, 1000, 2000, 3000]
        
        parameters = dict(#pca__n_components = n_components,
                      dec_tree__criterion = criterion,
                      dec_tree__max_depth = max_depth,
                      dec_tree__n_estimators = n_estimators)
        
        print('Best Criterion:', rf_optimized.best_estimator_.get_params()['dec_tree__criterion'])
        print('Best max_depth:', rf_optimized.best_estimator_.get_params()['dec_tree__max_depth'])
        print('Best n_estimators:', rf_optimized.best_estimator_.get_params()['dec_tree__n_estimators'])
        print('Best Number Of Components:', rf_optimized.best_estimator_.get_params()['pca__n_components'])
        print(); print(rf_optimized.best_estimator_.get_params()['dec_tree'])
       
        
        rf_optimized = GridSearchCV(rf_optimized, parameters, verbose = 3)"""

        # Intialization of the optimized model
        rf_optimized = RandomForestClassifier(n_estimators = 1000, criterion = 'gini')
        
        # Fit of the model
        rf_optimized.fit(X_train_processed, Y_train_processed)
        self.rf_optimized = rf_optimized
        
        # Verify that the model is correct
        # print('Cross-validation of the fit\n')
        # score = cross_val_score(rf_optimized, X_train_processed, Y_train_processed, cv = 10)
        # print('\nTest score : ', score.mean())

        # Plot the learning curve
        #self.plot_learning_curve(self.rf_optimized, X_train_processed, Y_train_processed)

        

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        # TODO: make predictions on X_data and return them

        # Convert 'white' and 'red' into binary values
        X_test_processed = self.X_data_preprocessing(X_data)
        
        # Remove the id's of the train data 
        X_test_processed = [elem[1:] for elem in X_test_processed]
                
        # Do the prediction
        rf_optimized = self.rf_optimized
        pred_RF = rf_optimized.predict(X_test_processed)
        
        
        # Format the array so that there is no issue 
        # Round to be sure those are int
        for i in range(len(pred_RF)):
            pred_RF[i] = np.round(pred_RF[i])
          
        # Convert the result into int
        pred_RF = pred_RF.astype(np.int64)


        # Add the IDs to the result
        pred = []
        for i in range(len(pred_RF)):
            pred.append([i, pred_RF[i]])

        return pred