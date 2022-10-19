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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



class MyWineTester(WineTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.rf = RandomForestClassifier(n_estimators = 1000)
        #self.rf = ExtraTreesClassifier(n_estimators = 1000)
        self.rf_optimized = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')

    
    def X_data_preprocessing(self, X_data):
        X_processed = X_data.copy()
        
        # Changer 'color' == 'red' en '1' et 'color' == 'white' en '0'
        N = len(X_data)
        for i in range (N):
            if X_data[i][1] == 'red': 
                X_data[i][1] = '1'
                X_data[i][2] = '0'
            if X_data[i][1] == 'white': 
                X_data[i][1] = '0'
                X_data[i][2] = '2'
                
        return X_processed
    
    
    def data_processing(self, X_data):
        X_data2 = X_data.copy()
        for column in X_data:
            X_data2 = X_data2[np.abs(X_data2[column] - X_data2[column].mean())<=(5*X_data2[column].std())]
            
        return X_data2


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
        X_train_processed = self.X_data_preprocessing(X_train)
        
        # Remove the id's of the train data 
        X_train_processed = [elem[1:] for elem in X_train]
        Y_train_processed = [elem[1] for elem in y_train]
        
        
        
        print('First fit\n')
        #rf = self.rf
        #rf.fit(X_train_processed, Y_train_processed)
        
        print('Cross-validation of the first fit\n')
        #score = cross_val_score(rf, X_train_processed, Y_train_processed, cv = 2)
        #print('\nTest score : ', score.mean())
        
        print('Second fit\n')
        
        
        # Initialize objects
        std_slc = StandardScaler()
        std_slc.fit_transform(X_train_processed)
        
        rf_optimized = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
        
        n_components = [12]
        criterion = ['entropy']
        max_depth = [None]
        n_estimators = [1000]
        
        
        parameters = dict(#pca__n_components = n_components,
                      dec_tree__criterion = criterion,
                      dec_tree__max_depth = max_depth,
                      dec_tree__n_estimators = n_estimators)
        
        #rf_optimized = GridSearchCV(pipe, parameters, verbose = 3)
        rf_optimized.fit(X_train_processed, Y_train_processed)
        
        print('Cross-validation of the second fit\n')
        score = cross_val_score(rf_optimized, X_train_processed, Y_train_processed, cv = 2)
        print('\nTest score : ', score.mean())
        
        self.rf_optimized = rf_optimized
        
        # print('Best Criterion:', rf_optimized.best_estimator_.get_params()['dec_tree__criterion'])
        # print('Best max_depth:', rf_optimized.best_estimator_.get_params()['dec_tree__max_depth'])
        # print('Best n_estimators:', rf_optimized.best_estimator_.get_params()['dec_tree__n_estimators'])
        # print('Best Number Of Components:', rf_optimized.best_estimator_.get_params()['pca__n_components'])
        # print(); print(rf_optimized.best_estimator_.get_params()['dec_tree'])
       
        

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
        X_test_processed = self.X_data_preprocessing(X_data)
        
        # Remove the id's of the train data 
        X_test_processed = [elem[1:] for elem in X_test_processed]
        
        # Faire la prédiction
        rf_optimized = self.rf_optimized
        pred_RF = rf_optimized.predict(X_test_processed)
        
        for i in range(len(pred_RF)):
            pred_RF[i] = np.round(pred_RF[i])
            
        pred_RF = pred_RF.astype(np.int64)

        pred = []
        for i in range(len(pred_RF)):
            pred.append([i, pred_RF[i]])
        #print('predict', pred)
        return pred
