from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
import joblib
import os
import pandas as pd
from sklearn.metrics import classification_report,make_scorer, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(\
            os.path.abspath(__file__)),"../.."))

RAW_DATA_PATH = os.path.join(ROOT_PATH,"raw_data")
ML_MODEL_PATH = os.path.join(RAW_DATA_PATH,"models")

DEFAULT_MODEL_DICT = {
    # "Logistic Regression": [{
    #     'classifier': [LogisticRegression(max_iter=10000)],
    #      "hyperparameter":{
    #     'classifier__penalty': ['l2'],
    #     'classifier__C': [0.1, 1.0, 10.0]
    #      }
    # }],
    # "SVC":[ {
    #     'classifier': [SVC()],
    #      "hyperparameter":{
    #     'classifier__kernel': ['linear', 'rbf'],
    #     'classifier__C': [0.1, 1.0, 10.0]
    #      }
    # }],
    "Random Forest": [{
        'classifier': [RandomForestClassifier()],
         "hyperparameter":{
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
         }
    }],
    # "KNeighborsClassifier": [{
    #     'classifier': [KNeighborsClassifier()],
    #      "hyperparameter":{
    #     'classifier__n_neighbors': [3, 5, 7]
    #      }
    # }],
    # "Decision Tree Classifier": [{

    #     'classifier': [DecisionTreeClassifier()],
    #      "hyperparameter":{
    #     'classifier__max_depth': [None, 10, 20]
    #      }
    # }],
    # "Gradient Boosting Classifier": [{
    #     'classifier': [GradientBoostingClassifier()],
    #      "hyperparameter":{
    #     'classifier__n_estimators': [50, 100],
    #     'classifier__learning_rate': [0.01, 0.1, 1.0]
    #      }
    # }],
    # "MLP Classifier": [{
    #     'classifier': [MLPClassifier(max_iter=10000)],
    #     "hyperparameter":{
    #     'classifier__hidden_layer_sizes': [(50,), (100,), (100, 100)],
    #     'classifier__max_iter': [10000]  # Increased max_iter
    #     }
    # }]
}

def ml_model_selection(X,y,param_grid=DEFAULT_MODEL_DICT):
    scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
    }
    for key,value in param_grid.items():
        pipeline = Pipeline([
            ('classifier', value[0]["classifier"][0])  # Placeholder for the classifier
        ])
        grid_param = value[0]["hyperparameter"]
        grid_search = GridSearchCV(pipeline,grid_param,cv=5,scoring=scoring,refit="accuracy",
                                n_jobs=-1)

        grid_search.fit(X,y)

        best_model = grid_search.best_estimator_


        results = grid_search.cv_results_

        y_pred = best_model.predict(X)

        # Step 3: Calculate precision, recall, and F1 score
        report = classification_report(y, y_pred, output_dict=True)
        best_index = grid_search.best_index_

        # Display results for the best model
        print(f"Best Model Performance for {key}:")
        print(f"Best Parameters: {grid_search.best_params_}")

        print("\nBest Model Metrics:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Mean Accuracy: {results['mean_test_accuracy'][best_index]:.4f}")
        print(f"Best Mean Precision: {results['mean_test_precision'][best_index]:.4f}")
        print(f"Best Mean Recall: {results['mean_test_recall'][best_index]:.4f}")
        print(f"Best Mean F1 Score: {results['mean_test_f1'][best_index]:.4f}")
        print("--------------------------------------------------------")
        print("--------------------------------------------------------")
        print("\n\n\n")

        model_path = os.path.join(ML_MODEL_PATH,f"{key}_model.pkl")
        with open(model_path, 'wb') as f:
            print(f"✅ Saving to {model_path}")
            pickle.dump(grid_search.best_estimator_, f)

        return grid_search.best_estimator_


def dl_model_selection(X,y):
    input_dim =X.shape[1]
    def create_model(nodes_layer1=32, nodes_layer2=16, num_layers=2, activation='relu', optimizer='adam',input_dim=27):
        model = Sequential()
        model.add(Dense(nodes_layer1, activation=activation, input_shape=(input_dim,)))

        for _ in range(num_layers - 1):
            model.add(Dense(nodes_layer2, activation=activation))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    model = KerasClassifier(
        model=create_model,
        epochs=10,
        batch_size=32,
        verbose=0
    )
    param_grid = {
        'batch_size': [10, 20],
        'epochs': [10, 20],
        'model__nodes_layer1': [16, 32, 64],
        'model__nodes_layer2': [8, 16, 32],
        'model__num_layers': [2, 3, 4],
        'model__activation': ['relu']
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X, y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


    best_model = grid_result.best_estimator_.model_
    model_path = os.path.join(ML_MODEL_PATH,f"DL_model.h5")
    best_index = grid_result.best_index_
    best_model.save(model_path)

    return best_model
