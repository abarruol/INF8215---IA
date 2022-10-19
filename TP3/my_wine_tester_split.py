"""
Team:
<<<<< Les Retardataires >>>>>
Authors:
<<<<<  Barruol Augustin - 2161214 >>>>>
<<<<< DE LA GRANDIERE Bathylle - MATRICULE #2 >>>>>
"""

# cd Desktop/Cours/INF8215\ -\ IA/tp3
# python main.py --train_file data/train.csv --test_file data/test_public.csv --prediction_file data/sample_submission.csv


from wine_testers import WineTester

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error
from sklearn import model_selection



class MyWineTester(WineTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.rf = RandomForestClassifier(n_estimators = 2000, criterion = 'entropy')
        self.rf_red = RandomForestClassifier(n_estimators = 5000, criterion = 'entropy')
        self.rf_white = RandomForestClassifier(n_estimators = 5000, criterion = 'entropy')
    
    def X_data_preprocessing(self, X_data):
        X_processed = X_data.copy()
        
        # Changer 'color' == 'red' en '1' et 'color' == 'white' en '0'
        N = len(X_data)
        for i in range (N):
            if X_data[i][1] == 'red': X_data[i][1] = '1'
            if X_data[i][1] == 'white': X_data[i][1] = '0'

        return X_processed
    
    def remove_outliers(self, X_data):
        X_cleaned = []
        X_cleaned = X_data[:]
        for column in X_cleaned:
            X_cleaned = X_cleaned[np.abs(X_cleaned[column] - X_cleaned[column].mean())<=(5*X_cleaned[column].std())]

        return X_cleaned
    
    def split_red_white(self, X_data, Y_data = None, value = 'red'):
        # Keep only the wines of 'color' == value and remove the column 'color'
        X_splitted, Y_splitted = [], []
        
        for i in range(len(X_data)):
            if X_data[i][1] == value:
                X_splitted.append(X_data[i][0:1] + X_data[i][2:])
                if Y_data != None: 
                    Y_splitted.append(Y_data[i])
            
        return X_splitted, Y_splitted
    

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
        
        # Convert white and red into binary values
        #X_train_processed = self.X_data_preprocessing(X_train)
        X_train_red, Y_train_red = self.split_red_white(X_train, y_train, value = 'red')
        X_train_white, Y_train_white = self.split_red_white(X_train, y_train, value = 'white')
        X_train_red = self.X_data_preprocessing(X_train_red)
        X_train_white = self.X_data_preprocessing(X_train_white)
        
        # Remove the id's of the train data 
        X_train_red_processed = [elem[1:] for elem in X_train_red]
        Y_train_red_processed = [elem[1] for elem in Y_train_red]
        
        X_train_white_processed = [elem[1:] for elem in X_train_white]
        Y_train_white_processed = [elem[1] for elem in Y_train_white]
                  
        std_slc_red = StandardScaler()
        std_slc_red.fit_transform(X_train_red_processed)
        
        std_slc_white = StandardScaler()
        std_slc_white.fit_transform(X_train_white_processed)
        
        print('Start of first fit : without data processing')
        rf_red = self.rf_red
        rf_red.fit(X_train_red_processed, Y_train_red_processed)
        
        rf_white = self.rf_white
        rf_white.fit(X_train_white_processed, Y_train_white_processed)
        
        print('End of first fit : without data processing')
        
        
        print('Cross-validation of the first fit\n')
        cross_val_score_red = model_selection.cross_val_score(rf_red, X_train_red_processed, Y_train_red_processed, cv = 2)
        print('\nTest score red : ', cross_val_score_red.mean())
        
        cross_val_score_white = model_selection.cross_val_score(rf_white, X_train_white_processed, Y_train_white_processed, cv = 2)
        print('\nTest score white : ', cross_val_score_white.mean())
        
        score = (len(X_train_red_processed)*cross_val_score_red.mean() + len(X_train_white_processed)*cross_val_score_white.mean()) / len(X_train)
        print('\nTest score : ', score.mean())
        print('Remove outliers...')
        #X_train_red_processed = self.remove_outliers(X_train_red_processed)
        #print(len(X_train_red_processed))
        

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

        # Changer red en (1,0) et white en (0,1)
        X_test_red, _ = self.split_red_white(X_data, value = 'red')
        X_test_white, _ = self.split_red_white(X_data, value = 'white')
        
        X_test_red = self.X_data_preprocessing(X_test_red)
        X_test_white = self.X_data_preprocessing(X_test_white) 
        
        # Remove the id's of the train data 
        X_test_red_processed = [elem[1:] for elem in X_test_red]
        X_test_white_processed = [elem[1:] for elem in X_test_white]
        
        # Remove attrivutes not relevant for each wine
        X_test_white_processed = np.delete(X_test_white_processed, 11, axis=1)
        X_test_red_processed = np.delete(X_test_red_processed, 0, axis=1)

        
        # Faire la prédiction
        rf_red = self.rf_red
        pred_RF_red = rf_red.predict(X_test_red_processed)
        
        rf_white = self.rf_white
        pred_RF_white = rf_white.predict(X_test_white_processed)
        
        for i in range(len(pred_RF_red)):
            pred_RF_red[i] = np.round(pred_RF_red[i])
            
        for i in range(len(pred_RF_white)):
            pred_RF_white[i] = np.round(pred_RF_white[i])
            
        pred_RF_red = pred_RF_red.astype(np.int64)
        pred_RF_white = pred_RF_white.astype(np.int64)
            
        pred = []
        index_red = 0
        index_white = 0
        for i in range(len(X_data)):
            if X_data[i][1] == 'red':
                pred.append([i, pred_RF_red[index_red]])
                index_red += 1
            if X_data[i][1] == 'white':
                pred.append([i, pred_RF_white[index_white]])
                index_white += 1
                
        return pred
