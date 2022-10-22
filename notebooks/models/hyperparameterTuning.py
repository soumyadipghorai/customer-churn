from analyseModel import results
from sklearn.model_selection import GridSearchCV

def hyperparamTraining(model, x_train, y_train, x_test, y_test, param_grid, of_type = None, columns = None):
    gridSearchmodel = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        n_jobs = -1,
        refit = True,
        verbose = 4,
        cv = 10,
        return_train_score = True
    )
    
    gridSearchmodel.fit(x_train, y_train)
    
    # print best parameter after tuning
    print(gridSearchmodel.best_params_)
    
    # print how our model looks after hyper-parameter tuning
    print(gridSearchmodel.best_estimator_)
    
    finalModel = gridSearchmodel.best_estimator_
    
    results(finalModel, x_train, y_train, x_test, y_test, of_type, columns)

    return finalModel