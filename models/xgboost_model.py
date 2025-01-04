import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error

def train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test, feature_names, n_trials=100):
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_names)
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  
            'device': 'cuda',   
            'eta': trial.suggest_float('eta', 0.001, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'lambda': trial.suggest_float('lambda', 0.0, 2.0),
            'alpha': trial.suggest_float('alpha', 0.0, 2.0),
            'nthread': -1,
            'seed': 42
        }
        
        num_round = 100
        model = xgb.train(params, dtrain, num_round)
        predictions = model.predict(dtest)
        rmse = root_mean_squared_error(y_test, predictions)
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update({
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror'
    })
    
    # Train final model with best parameters
    model = xgb.train(best_params, dtrain, 100)
    
    return model, best_params