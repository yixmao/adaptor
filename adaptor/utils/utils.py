import os
import argparse
import time
import random
import pickle
import joblib
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression as LR

    
# Calculate the performance metrics
def metric_calc(pred, target, verbose):
    """
    Computes various performance metrics for model evaluation.

    Parameters:
    - pred (np.ndarray): Model-predicted response values.
    - target (np.ndarray): True target values.
    - verbose (bool): Whether to print the computed metrics.

    Returns:
    - scorr (float): Spearman correlation coefficient.
    - r2 (float): R-squared (coefficient of determination).
    - rmse (float): Root Mean Squared Error (RMSE).
    - nrmse (float): Normalized RMSE (NRMSE), computed as RMSE divided by the standard deviation of the target.
    - pearson_corr (float): Pearson correlation coefficient.
    - bias (float): Bias in prediction.
    """
    scorr, pvalue = spearmanr(target, pred)
    r2 = r2_score(target, pred)
    rmse = mean_squared_error(target, pred, squared=False)
    std_i = np.std(target)
    nrmse = rmse / std_i
    pearson_corr, pearson_pvalue = pearsonr(pred, target)
    bias = Bias_Calc(pred, target).item()
    if verbose:
        print('Spearman: '+str(scorr))
        print('R2: '+str(r2))
        print('RMSE: '+str(rmse))
        print('NRMSE: '+str(nrmse))
        print('PCC: ' +str(pearson_corr))
        print('Bias: ' +str(bias))

    return scorr, r2, rmse, nrmse, pearson_corr, bias

def Bias_Calc(pred, target):
    """
    Computes the bias using a linear regression model.

    Parameters:
    - pred (np.ndarray): Predicted values.
    - target (np.ndarray): True target values.

    Returns:
    - bias (float): Bias in prediction.

    This function fits a simple linear regression model with target values as the independent variable
    and prediction errors as the dependent variable to estimate bias.
    """
    error = target - pred
    target = target.reshape(len(target),1)
    error = error.reshape(len(error),1)
    
    reg = LR().fit(target, error)
    bias = reg.coef_[0]
    
    return bias

def stack_metrics(pred, target, metrics, step, verbose):
    """
    Computes and stores performance metrics for a given prediction step.

    Parameters:
    - pred (np.ndarray): Predicted response values.
    - target (np.ndarray): True target values.
    - metrics (list): List to store computed metric results.
    - step (str or int): Step identifier for the metrics log.

    Returns:
    - metrics (list): Updated list containing performance metrics.
    """
    scorr, r2, rmse, nrmse, pearson_corr, bias = metric_calc(pred, target, verbose)
    # store the test performance
    metrics.append({
        'Step': step, 
        'Spearman_Corr': scorr,
        'R2_Score': r2,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'Pearson_Corr': pearson_corr,
        'Bias': bias
    })
    return metrics

# stack different models with a LR model
def stack_models(preds_val, preds_test, target, val_idx):
    """
    Combines predictions from multiple models using a linear regression stacking approach.

    Parameters:
    - preds_val (list of np.ndarray): List of model predictions for the validation set.
    - preds_test (list of np.ndarray): List of model predictions for the test set.
    - target (pd.DataFrame): True target values.
    - val_idx (list): Indices of the validation samples.

    Returns:
    - pred_test (np.ndarray): Final stacked predictions for the test set.
    - train_time (float): Time taken to train the stacking model.
    - test_time (float): Time taken to make predictions using the stacking model.
    """
    start_train_time = time.time() 
    preds_val_np = np.array(preds_val).T
    # Create and fit the linear model on the validation predictions
    stacking_model = LR(n_jobs=-1)
    stacking_model.fit(preds_val_np, target.loc[val_idx].values)
    train_time = time.time() - start_train_time

    # Use the stacking model to combine predictions on the test set
    start_test_time = time.time()
    preds_test_np = np.array(preds_test).T
    pred_test = stacking_model.predict(preds_test_np)
    test_time = time.time() - start_test_time
    return pred_test, train_time, test_time

def save_results(results_folder, output_suffix, test_metrics, preds_test, preds_val, preds_val_stack, indices, models, args):
    # Save the variables using pickle
    with open(f"{results_folder}/test_metrics_{output_suffix}.pkl", "wb") as f:
        pickle.dump(test_metrics, f)
    with open(f"{results_folder}/preds_test_{output_suffix}.pkl", "wb") as f:
        pickle.dump(preds_test, f)
    with open(f"{results_folder}/preds_val_{output_suffix}.pkl", "wb") as f:
        pickle.dump(preds_val, f)
    with open(f"{results_folder}/indices_{output_suffix}.pkl", "wb") as f:
        pickle.dump(indices, f)
    if args.integrate_method == 'stack':
        with open(f"{results_folder}/preds_val_stack_{output_suffix}.pkl", "wb") as f:
            pickle.dump(preds_val_stack, f)
    # save the model
    if args.save_mdls is True:
        joblib.dump(models, f"{results_folder}/models_{output_suffix}.joblib")
    # otherwise the models are too large

def load_results(results_folder, output_suffix): 
    # Load the variables using pickle
    with open(f"{results_folder}/test_metrics_{output_suffix}.pkl", "rb") as f:
        test_metrics = pickle.load(f)
    with open(f"{results_folder}/preds_test_{output_suffix}.pkl", "rb") as f:
        preds_test = pickle.load(f)
    with open(f"{results_folder}/preds_val_{output_suffix}.pkl", "rb") as f:
        preds_val = pickle.load(f)
    with open(f"{results_folder}/indices_{output_suffix}.pkl", "rb") as f:
        indices = pickle.load(f)
    if os.path.exists(f"{results_folder}/preds_val_stack_{output_suffix}.pkl"):
        with open(f"{results_folder}/preds_val_stack_{output_suffix}.pkl", "rb") as f:
            preds_val_stack = pickle.load(f)
    else:
        preds_val_stack = None
    # load the models
    if os.path.exists(f"{results_folder}/models_{output_suffix}.joblib"):
        models = joblib.load(f"{results_folder}/models_{output_suffix}.joblib")
    else:
        models = None
    return test_metrics, preds_test, preds_val, preds_val_stack, indices, models


def normalize_data(data, means=None, stds=None):
    """
    Normalizes data using z-score normalization.

    Parameters:
    - data (np.ndarray): Data to be normalized.
    - means (np.ndarray, optional): Mean values for each feature. If None, they are computed from `data`.
    - stds (np.ndarray, optional): Standard deviation values for each feature. If None, they are computed from `data`.

    Returns:
    - normalized_data (np.ndarray): Normalized data.
    - means (np.ndarray): Mean values used for normalization.
    - stds (np.ndarray): Standard deviation values used for normalization.
    """
    # Calculate the mean and standard deviation of each column
    if means is None:
        means = np.mean(data, axis=0)
    if stds is None:
        stds = np.std(data, axis=0)
    
    # Avoid division by zero in case of zero standard deviation
    stds[stds == 0] = 1
    
    # Normalize the data
    normalized_data = (data - means) / stds
    
    return normalized_data, means, stds

def denormalize_data(normalized_data, means, std):
    """
    Reverts normalized data back to its original scale.

    Parameters:
    - normalized_data (np.ndarray): Data that was normalized.
    - means (np.ndarray): Mean values used for normalization.
    - std (np.ndarray): Standard deviation values used for normalization.

    Returns:
    - data (np.ndarray): Denormalized data.

    This function reverses the normalization transformation by multiplying with the standard deviation
    and adding back the mean.
    """
    data = normalized_data*std+means
    return data

def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    Parameters:
    - seed (int): Seed value for random number generators.
    """
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed


def str2bool(v):
    """
    Converts a string representation of a boolean to a boolean value.

    Parameters:
    - v (str or bool): Input value (string or boolean).

    Returns:
    - bool: Converted boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')