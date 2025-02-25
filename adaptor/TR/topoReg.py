import time
import random
import warnings
import numpy as np
import pandas as pd
# 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
#
from scipy.optimize import minimize
#
from adaptor.TR.distance import calculate_distance_parallel
from adaptor.utils.utils import stack_metrics, stack_models

# TopoReg
def TopoReg(distance_x, distance_y, target, train_idx, test_idx, args):
    """
    Train and evaluate Topological Regression (TopoReg) models using structure and response distance matrices.

    Parameters:
    - distance_x (pd.DataFrame): Structure (input) distance matrix.
    - distance_y (pd.DataFrame): Response (output) distance matrix.
    - target (pd.DataFrame): Target/reponse values.
    - train_idx (list): Indices of training samples.
    - test_idx (list): Indices of test samples.
    - args (Namespace): A configuration object containing model parameters.

    Returns:
    - metrics (list of dicts): Performance metrics at each step, including 'Spearman_Corr', 'R2_Score', 'RMSE', 'NRMSE', 'Pearson_Corr', 'Bias', 'Training_Time' and 'Testing_Time'.
    - preds_test (list of np.ndarray): Predictions for the test set at each step.
    - preds_val (list of np.ndarray or None): Predictions for the validation set at each step, if applicable.
    - preds_val_stack (list of np.ndarray or None): Predictions for validation samples for stacking, if applicable.
    - pred_test (np.ndarray): Final predictions for the test set.
    - anchors_idx_x_all (list of lists): Selected structure anchor indices at each step.
    - anchors_idx_y (list): Selected response anchor indices.
    - models (list): Models trained at each step, only stored when args.save_mdls is True.
    """

    # randomly split validation samples as needed
    if args.val_frac <=0:
        val_idx = None
    else:
        train_idx, val_idx = train_test_split(train_idx, test_size=args.val_frac, random_state=args.seed)
    if args.integrate_method == 'stack':
        if val_idx is not None:
            val_idx, val_idx_stack = train_test_split(val_idx, test_size=0.5, random_state=args.seed)
        else:
            train_idx, val_idx_stack = train_test_split(train_idx, test_size=0.1, random_state=args.seed)
    else:
        val_idx_stack = None
    # Determin the number of structure anchors. For adaptive anchor selection, it is the number of structure anchors selected at each step. 
    if args.random_anchor_perc is False:
        if args.num_anchors_x<=1: # percentage 
            num_anchors = int(np.ceil(args.num_anchors_x * len(train_idx)))
        else: # number
            num_anchors = int(args.num_anchors_x)
    else: # random percent of structure anchors for the ensemble approach
        anchor_percentages = np.random.normal(args.mean_anchor_percentage,args.std_anchor_percentage,args.num_steps)
        # Clip anchor percentages to assert validity and prevent under/overfitting
        anchor_percentages[anchor_percentages<args.min_anchor_percentage]=args.min_anchor_percentage
        anchor_percentages[anchor_percentages>args.max_anchor_percentage]=args.max_anchor_percentage
        num_anchors = int(np.ceil(anchor_percentages[0] * len(train_idx)))
    
    # select the first set of anchors randomly - same for all methods
    anchors_idx_x = random.sample(train_idx, num_anchors)
    anchors_idx_x_all = [] # store the selected anchors

    # select the response anchors
    if args.anchors_y_sel == 'cluster': # use K-means clustering to select the response anchors
        anchors_idx_y = get_cluster_centers(target, train_idx, args.num_anchors_y)
        distance_y = distance_y.loc[:, anchors_idx_y]
    elif args.anchors_y_sel == 'same': # 'same' as the structure anchors
        anchors_idx_y = anchors_idx_x
    # initialization
    preds_test = []
    preds_val = []
    preds_val_stack = []
    metrics = []
    models = []
    total_train_time = 0
    total_test_time = 0    
    # for early stopping
    best_nrmse = float('inf')  # Initialize the best NRMSE to infinity
    wait = 0  # Counter for how many steps we've waited without improvement
    ## Start interation
    for i in range(args.num_steps):
        # store the selected anchors
        anchors_idx_x_all.append(anchors_idx_x)
        # Get the validation distances
        if val_idx is None:
            pred_val = None
            dist_val = None
        else:
            dist_val = distance_x.loc[val_idx, anchors_idx_x]
        if val_idx_stack is None:
            pred_val_stack = None
            dist_val_stack = None
        else:
            dist_val_stack = distance_x.loc[val_idx_stack, anchors_idx_x]
        # Sample training and testing distances
        dist_x_train = distance_x.loc[train_idx, anchors_idx_x]
        dist_y_train = distance_y.loc[train_idx, anchors_idx_y]
        dist_test = distance_x.loc[test_idx, anchors_idx_x]
        # train the model
        mdl, train_time = mdl_train(dist_x_train, dist_y_train, args)
        total_train_time += train_time
        # predict the test samples
        pred_test, test_time = mdl_pred(mdl, dist_test, target.loc[anchors_idx_y], anchors_idx_y, args)
        total_test_time += test_time
        # predict the validation samples
        if val_idx is not None:
            pred_val, train_time = mdl_pred(mdl, dist_val, target.loc[anchors_idx_y], anchors_idx_y, args)
            total_train_time += train_time
        if val_idx_stack is not None:
            pred_val_stack, train_time = mdl_pred(mdl, dist_val_stack, target.loc[anchors_idx_y], anchors_idx_y, args)
            total_train_time += train_time
        # store the models
        if args.save_mdls: # save mdls or not
            models.append(mdl)

        # Store the predicted responses for test and validation sets
        preds_test.append(pred_test)
        preds_val.append(pred_val)
        preds_val_stack.append(pred_val_stack)

        # Evaluate the performance of the single model
        if args.verbose:
            print(f'Single model performance:')
        if pred_val is not None:
            if args.verbose:
                print(f'Step {i+1}, val performance:')
            stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, f"singleMdl_val_step{i+1}", args.verbose)
            val_nrmse = metrics[-1]['NRMSE']
        if args.verbose:
            print(f'Step {i+1}, test performance:')
        stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, f"singleMdl_test_step{i+1}", args.verbose)

        # Ensemble average 
        if args.integrate_method == 'ensemble':  
            pred_test = np.array(preds_test).mean(axis=0)
            # evaluation
            if args.verbose:
                print(f'Ensemble performance:')
            if pred_val is not None:
                pred_val = np.array(preds_val).mean(axis=0)
                if args.verbose:
                    print(f'Step {i+1}, val performance:')
                stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, f"ensemble_val_step{i+1}", args.verbose)
                val_nrmse = metrics[-1]['NRMSE']
            if args.verbose:
                print(f'Step {i+1}, test performance:')
            stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, f"ensemble_test_step{i+1}", args.verbose)
        
        # Stack the models
        if args.integrate_method == 'stack':
            # stack the results 
            pred_test, _, _ = stack_models(preds_val_stack, preds_test, target, val_idx_stack)
            pred_test = pred_test.flatten()
            # evaluation
            if args.verbose:
                print(f'Stacking performance:')
            if pred_val is not None:
                pred_val, _, _ = stack_models(preds_val_stack, preds_val, target, val_idx_stack)
                if args.verbose:
                    print(f'Step {i+1}, val performance:')
                stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, f"stack_val_step{i+1}", args.verbose)
                val_nrmse = metrics[-1]['NRMSE']
            if args.verbose:
                print(f'Step {i+1}, test performance:')
            stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, f"stack_test_step{i+1}", args.verbose)

        # Store time
        metrics[-1]['Training_Time'] = total_train_time
        if args.integrate_method in ['ensemble', 'stack']:
            metrics[-1]['Testing_Time'] = total_test_time
        else: # single model test does not rely one previous models
            metrics[-1]['Testing_Time'] = test_time

        # If it is not the last step
        if i+1 < args.num_steps:
            # check for early stop
            if pred_val is not None:
                current_nrmse = val_nrmse
                if current_nrmse < best_nrmse - args.min_delta:
                    best_nrmse = current_nrmse
                    wait = 0  # Reset the counter if there's an improvement
                else:
                    wait += 1  # Increment the counter if no improvement

            # Check for early stopping
            if ((wait >= args.patience) & args.early_stop):
                if args.verbose:
                    print(f"Early stopping at step {i+1}. Best NRMSE: {best_nrmse:.4f}")
                break
                
            # select the next groups of anchors
            if args.random_anchor_perc is True: # random percent of anchors
                num_anchors = int(np.ceil(anchor_percentages[i+1] * len(train_idx)))
            # anchor selection
            if args.anchor_x_sel == 'adaptive': # adaptive anchor selection
                # get a new list of train_idx that removes the elements in anchors_idx_x
                train_idx_new = [idx for idx in train_idx if idx not in anchors_idx_x]
                # get preds for these samples
                pred_train, train_time = mdl_pred(mdl, dist_x_train.loc[train_idx_new], target.loc[anchors_idx_y], anchors_idx_y, args)
                total_train_time += train_time
                # select the training samples showing the largest abs prediction errors as the anchor points
                anchors_idx_x_new = anchor_select_pred_train_err(target, train_idx_new, pred_train, num_anchors)
            elif args.anchor_x_sel == 'random': # random selection
                if args.append_anchors is True:
                    # get a new list of train_idx that removes the elements in anchors_idx_x
                    train_idx_new = [idx for idx in train_idx if idx not in anchors_idx_x]
                    anchors_idx_x_new = random.sample(train_idx_new, num_anchors)
                else:
                    anchors_idx_x_new = random.sample(train_idx, num_anchors)
                    

            # Append structure anchors or not
            if args.append_anchors is True:
                if args.verbose:
                    print('Appending Anchors')
                num_anchors_old = len(anchors_idx_x)
                anchors_idx_x = list(set(anchors_idx_x_new + anchors_idx_x))
                num_anchors_new = len(anchors_idx_x)
                if args.verbose:
                    print(f'# of anchors: {num_anchors_new}; '
                        f'%: {num_anchors_new / len(train_idx) * 100:.2f}; '
                        f'# of new anchors: {num_anchors_new - num_anchors_old}; '
                        f'{(num_anchors_new - num_anchors_old) / num_anchors * 100:.2f}')
            else:
                anchors_idx_x = anchors_idx_x_new
                if args.verbose:
                    print(f'# of anchors:{len(anchors_idx_x)}')
            # Set response anchors
            if args.anchors_y_sel == 'same':
                anchors_idx_y = anchors_idx_x

    # record the final results
    if pred_val is not None:
        stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, 'val_final', args.verbose)
    print('Final test performance:')
    stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, 'test_final', verbose=True)
    # store time
    metrics[-1]['Training_Time'] = total_train_time
    if args.integrate_method in ['ensemble', 'stack']:
        metrics[-1]['Testing_Time'] = total_test_time
    else: # single model test does not rely one previous models
        metrics[-1]['Testing_Time'] = test_time
    return metrics, preds_test, preds_val, preds_val_stack, pred_test, anchors_idx_x_all, anchors_idx_y, models

# TopoReg with desciptors
def TopoReg_desc(desc, target, train_idx, test_idx, args):
    """
    Train and evaluate Topological Regression (TopoReg) models using descriptors and target values. It is recommended to calculate the distances first and use the function 'TopoReg'.

    Parameters:
    - desc (pd.DataFrame): Molecular descriptors used to calculate the structure distances.
    - target (pd.DataFrame): Target/reponse values.
    - train_idx (list): Indices of training samples.
    - test_idx (list): Indices of test samples.
    - args (Namespace): A configuration object containing model parameters.

    Returns:
    - metrics (list of dicts): Performance metrics at each step, including 'Spearman_Corr', 'R2_Score', 'RMSE', 'NRMSE', 'Pearson_Corr', 'Bias', 'Training_Time' and 'Testing_Time'.
    - preds_test (list of np.ndarray): Predictions for the test set at each step.
    - preds_val (list of np.ndarray or None): Predictions for the validation set at each step, if applicable.
    - preds_val_stack (list of np.ndarray or None): Predictions for validation samples for stacking, if applicable.
    - pred_test (np.ndarray): Final predictions for the test set.
    - anchors_idx_x_all (list of lists): Selected structure anchor indices at each step.
    - anchors_idx_y (list): Selected response anchor indices.
    - models (list): Models trained at each step, only stored when args.save_mdls is True.
    """

    # randomly split validation samples as needed
    if args.val_frac <=0:
        val_idx = None
    else:
        train_idx, val_idx = train_test_split(train_idx, test_size=args.val_frac, random_state=args.seed)
    if args.integrate_method == 'stack':
        if val_idx is not None:
            val_idx, val_idx_stack = train_test_split(val_idx, test_size=0.5, random_state=args.seed)
        else:
            train_idx, val_idx_stack = train_test_split(train_idx, test_size=0.1, random_state=args.seed)
    else:
        val_idx_stack = None

    # Determin the number of structure anchors. For adaptive anchor selection, it is the number of structure anchors selected at each step. 
    if args.random_anchor_perc is False:
        if args.num_anchors_x<=1: # percentage 
            num_anchors = int(np.ceil(args.num_anchors_x * len(train_idx)))
        else: # number
            num_anchors = int(args.num_anchors_x)
    else: # random percent of structure anchors for the ensemble approach
        anchor_percentages = np.random.normal(args.mean_anchor_percentage,args.std_anchor_percentage,args.num_steps)
        # Clip anchor percentages to assert validity and prevent under/overfitting
        anchor_percentages[anchor_percentages<args.min_anchor_percentage]=args.min_anchor_percentage
        anchor_percentages[anchor_percentages>args.max_anchor_percentage]=args.max_anchor_percentage
        num_anchors = int(np.ceil(anchor_percentages[0] * len(train_idx)))
    
    # select the first set of anchors randomly - same for all methods
    anchors_idx_x = random.sample(train_idx, num_anchors)
    anchors_idx_x_all = [] # store the selected anchors

    # select the response anchors
    if args.anchors_y_sel == 'cluster': # use K-means clustering to select the response anchors
        anchors_idx_y = get_cluster_centers(target, train_idx, args.num_anchors_y) 
    elif args.anchors_y_sel == 'same': # 'same' as the structure anchors
        anchors_idx_y = anchors_idx_x
    # initialization
    preds_test = []
    preds_val = []
    preds_val_stack = []
    metrics = []
    models = []
    total_train_time = 0
    total_test_time = 0    
    # for early stopping
    best_nrmse = float('inf')  # Initialize the best NRMSE to infinity
    wait = 0  # Counter for how many steps we've waited without improvement
    # Calculate the distances
    if args.verbose:
        print('Start calculating the distances')
    if val_idx is None:
        pred_val = None
        dist_val = None
    else:
        dist_val = calculate_distance_parallel(desc, val_idx, anchors_idx_x, args.distance, args.cpus)
    if val_idx_stack is None:
        pred_val_stack = None
        dist_val_stack = None
    else:
        dist_val_stack = calculate_distance_parallel(desc, val_idx_stack, anchors_idx_x, args.distance, args.cpus)
    # calculate training and testing distances
    dist_x_train = calculate_distance_parallel(desc, train_idx, anchors_idx_x, args.distance, args.cpus)
    dist_y_train = calculate_distance_parallel(target, train_idx, anchors_idx_y, 'euclidean', args.cpus)
    dist_test = calculate_distance_parallel(desc, test_idx, anchors_idx_x, args.distance, args.cpus)
    if args.verbose:
        print('Finished calculating the distances')
    for i in range(args.num_steps):
        # store the selected anchors
        anchors_idx_x_all.append(anchors_idx_x)
        # train the model
        mdl, train_time = mdl_train(dist_x_train, dist_y_train, args)
        total_train_time += train_time
        # predict the test samples
        pred_test, test_time = mdl_pred(mdl, dist_test, target.loc[anchors_idx_y], anchors_idx_y, args)
        total_test_time += test_time
        # predict the validation samples
        if val_idx is not None:
            pred_val, train_time = mdl_pred(mdl, dist_val, target.loc[anchors_idx_y], anchors_idx_y, args)
            total_train_time += train_time
        if val_idx_stack is not None:
            pred_val_stack, train_time = mdl_pred(mdl, dist_val_stack, target.loc[anchors_idx_y], anchors_idx_y, args)
            total_train_time += train_time
        # store the models
        if args.save_mdls: # save mdls or not
            models.append(mdl)

        # Store the predicted responses for test and validation sets
        preds_test.append(pred_test)
        preds_val.append(pred_val)
        preds_val_stack.append(pred_val_stack)

        # Evaluate the performance of the single model
        if args.verbose:
            print(f'Single model performance:')
        if pred_val is not None:
            if args.verbose:
                print(f'Step {i+1}, val performance:')
            stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, f"singleMdl_val_step{i+1}", args.verbose)
            val_nrmse = metrics[-1]['NRMSE']
        if args.verbose:
            print(f'Step {i+1}, test performance:')
        stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, f"singleMdl_test_step{i+1}", args.verbose)

        # Ensemble average 
        if args.integrate_method == 'ensemble':  
            pred_test = np.array(preds_test).mean(axis=0)
            # evaluation
            if args.verbose:
                print(f'Ensemble performance:')
            if pred_val is not None:
                pred_val = np.array(preds_val).mean(axis=0)
                if args.verbose:
                    print(f'Step {i+1}, val performance:')
                stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, f"ensemble_val_step{i+1}", args.verbose)
                val_nrmse = metrics[-1]['NRMSE']
            if args.verbose:
                print(f'Step {i+1}, test performance:')
            stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, f"ensemble_test_step{i+1}", args.verbose)
        
        # Stack the models
        if args.integrate_method == 'stack':
            # stack the results 
            pred_test, _, _ = stack_models(preds_val_stack, preds_test, target, val_idx_stack)
            pred_test = pred_test.flatten()
            # evaluation
            if args.verbose:
                print(f'Stacking performance:')
            if pred_val is not None:
                pred_val, _, _ = stack_models(preds_val_stack, preds_val, target, val_idx_stack)
                if args.verbose:
                    print(f'Step {i+1}, val performance:')
                stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, f"stack_val_step{i+1}", args.verbose)
                val_nrmse = metrics[-1]['NRMSE']
            if args.verbose:
                print(f'Step {i+1}, test performance:')
            stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, f"stack_test_step{i+1}", args.verbose)

        # Store time
        metrics[-1]['Training_Time'] = total_train_time
        if args.integrate_method in ['ensemble', 'stack']:
            metrics[-1]['Testing_Time'] = total_test_time
        else: # single model test does not rely one previous models
            metrics[-1]['Testing_Time'] = test_time
        
        # If it is not the last step
        if i+1 < args.num_steps:
            # check for early stop
            if pred_val is not None:
                current_nrmse = val_nrmse
                if current_nrmse < best_nrmse - args.min_delta:
                    best_nrmse = current_nrmse
                    wait = 0  # Reset the counter if there's an improvement
                else:
                    wait += 1  # Increment the counter if no improvement

            # Check for early stopping
            if ((wait >= args.patience) & args.early_stop):
                if args.verbose:
                    print(f"Early stopping at step {i+1}. Best NRMSE: {best_nrmse:.4f}")
                break
                
           # select the next groups of anchors
            if args.random_anchor_perc is True: # random percent of anchors
                num_anchors = int(np.ceil(anchor_percentages[i+1] * len(train_idx)))
            # anchor selection
            if args.anchor_x_sel == 'adaptive': # adaptive anchor selection
                # get a new list of train_idx that removes the elements in anchors_idx_x
                train_idx_new = [idx for idx in train_idx if idx not in anchors_idx_x]
                # get preds for these samples
                pred_train, train_time = mdl_pred(mdl, dist_x_train.loc[train_idx_new], target.loc[anchors_idx_y], anchors_idx_y, args)
                total_train_time += train_time
                # select the training samples showing the largest abs prediction errors as the anchor points
                anchors_idx_x_new = anchor_select_pred_train_err(target, train_idx_new, pred_train, num_anchors)
            elif args.anchor_x_sel == 'random': # random anchor selection
                if args.append_anchors is True:
                    # get a new list of train_idx that removes the elements in anchors_idx_x
                    train_idx_new = [idx for idx in train_idx if idx not in anchors_idx_x]
                    anchors_idx_x_new = random.sample(train_idx_new, num_anchors)
                else:
                    anchors_idx_x_new = random.sample(train_idx, num_anchors)

            # calculate the distances with the new anchors
            # validation samples
            if val_idx is not None:
                dist_val_new = calculate_distance_parallel(desc, val_idx, anchors_idx_x_new, args.distance, args.cpus)
            if val_idx_stack is not None:
                dist_val_stack_new = calculate_distance_parallel(desc, val_idx_stack, anchors_idx_x_new, args.distance, args.cpus)
            # training and testing samples
            dist_x_train_new = calculate_distance_parallel(desc, train_idx, anchors_idx_x_new, args.distance, args.cpus)
            dist_test_new = calculate_distance_parallel(desc, test_idx, anchors_idx_x_new, args.distance, args.cpus)

            # append anchors or not
            if args.append_anchors is True:
                if args.verbose:
                    print('Appending Anchors')
                num_anchors_old = len(anchors_idx_x)
                anchors_idx_x = list(set(anchors_idx_x_new + anchors_idx_x))
                num_anchors_new = len(anchors_idx_x)
                if args.verbose:
                    print(f'# of anchors: {num_anchors_new}; '
                        f'%: {num_anchors_new / len(train_idx) * 100:.2f}; '
                        f'# of new anchors: {num_anchors_new - num_anchors_old}; '
                        f'{(num_anchors_new - num_anchors_old) / num_anchors * 100:.2f}')

                # append the distance matrices
                if val_idx is not None:
                    dist_val = pd.concat([dist_val, dist_val_new], axis=1)
                if val_idx_stack is not None:
                    dist_val_stack = pd.concat([dist_val_stack, dist_val_stack_new], axis=1)
                dist_x_train = pd.concat([dist_x_train, dist_x_train_new], axis=1)
                dist_test = pd.concat([dist_test, dist_test_new], axis=1)
            else:
                anchors_idx_x = anchors_idx_x_new
                if args.verbose:
                    print(f'# of anchors:{len(anchors_idx_x)}')
                if val_idx is not None:
                    dist_val = dist_val_new
                if val_idx_stack is not None:
                    dist_val_stack = dist_val_stack_new
                dist_x_train = dist_x_train_new
                dist_test = dist_test_new

            # Update the response distances if the response anchors are the same as the structure anchors. Otherwise, no need to update the response distances.
            if args.anchors_y_sel == 'same':
                # calculate new dist_y
                dist_y_train_new = calculate_distance_parallel(target, train_idx, anchors_idx_x_new, 'euclidean', args.cpus)
                if args.append_anchors is True:
                    dist_y_train = pd.concat([dist_y_train, dist_y_train_new], axis=1)
                else:
                    dist_y_train = dist_y_train_new
                # update response anchors
                anchors_idx_y = anchors_idx_x

    # record the final results
    if pred_val is not None:
        stack_metrics(pred_val, target.loc[val_idx].values.flatten(), metrics, 'val_final', args.verbose)
    print('Final test performance:')
    stack_metrics(pred_test, target.loc[test_idx].values.flatten(), metrics, 'test_final', verbose=True)
    # store time
    metrics[-1]['Training_Time'] = total_train_time
    if args.integrate_method in ['ensemble', 'stack']:
        metrics[-1]['Testing_Time'] = total_test_time
    else: # single model test does not rely one previous models
        metrics[-1]['Testing_Time'] = test_time
    return metrics, preds_test, preds_val, preds_val_stack, pred_test, anchors_idx_x_all, anchors_idx_y, models

def get_cluster_centers(target, train_idx, num_clust):
    """
    Selects response anchors from training samples using K-Means clustering.

    Parameters:
    - target (pd.DataFrame): Target values of training samples.
    - train_idx (list): Indices of training samples.
    - num_clust (int): Number of clusters to form.

    Returns:
    - list: Indices of the training samples closest to each cluster center.
    """
    # apply mini-batch kmeans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        kmeans = MiniBatchKMeans(n_clusters=num_clust, batch_size=len(train_idx), n_init='auto')
        # kmeans.fit(target.loc[train_idx].values.reshape(-1, 1))
        kmeans.fit(target.loc[train_idx])
    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Get the cluster labels (the cluster each point is assigned to)
    cluster_labels = kmeans.labels_

    train_idx_np = np.array(train_idx)
    # Initialize a list to store the indices of the closest points
    center_indices = []

    # Loop over each cluster
    for cluster in range(kmeans.n_clusters):
        # Get the indices of the points in the current cluster
        cluster_points_idx = train_idx_np[np.where(cluster_labels == cluster)[0]]
        # Get the data points in the current cluster
        cluster_points = target.loc[cluster_points_idx].values
        # Get the center of the current cluster
        cluster_center = cluster_centers[cluster].reshape(1, -1)
        # Find the index of the point closest to the cluster center
        closest_idx_in_cluster, _ = pairwise_distances_argmin_min(cluster_center, cluster_points)
        # Map this index back to the original dataset's indices
        closest_idx = cluster_points_idx[closest_idx_in_cluster[0]]
        # Store the index of the closest point
        center_indices.append(closest_idx)

    return center_indices

def mdl_train(dist_x_train, dist_y_train, args):
    """
    Trains a linear model using the distance matrices.

    Parameters:
    - dist_x_train (pd.DataFrame): Structure distance matrix between training samples and structure anchors
    - dist_y_train (pd.DataFrame): Response distance matrix between training samples and response anchors
    - args: Namespace, contains model configuration (e.g., model type, ANN hyperparameters)

    Returns:
    - mdl: trained model
    - elapsed_time (float): time elapsed in training
    """
    start_time = time.time()
    # Modelling - start training
    if args.model == 'LR':
        mdl = LinearRegression(n_jobs=-1)
    elif args.model == 'LR_L2': 
        mdl = Ridge(alpha=args.ridge_alpha)
    else:
        raise ValueError("Unsupported model type. Choose 'LR' and 'LR_L2'.")
    mdl.fit(dist_x_train, dist_y_train)
    elapsed_time = time.time() - start_time
    return mdl, elapsed_time

def mdl_pred(mdl, dist_x, target, anchors_idx_y, args):
    """
    Predicts the reponses from the structure distances using a trained models.

    Parameters:
    - mdl: Trained linear model
    - dist_x (pd.DataFrame): Structure distance matrix between testing samples and response anchors
    - target (pd.DataFrame): Target values of the response anchors
    - anchors_idx_y (list): Indices of reponse anchors
    - args: Namespace, contains model configuration (e.g., model type, ANN hyperparameters)

    Returns:
    - pred (1D array): Predictions of the testing samples 
    - elapsed_time (float): time elapsed in testing
    """
    start_time = time.time()
    # get reponse distance estimations
    dist_array = mdl.predict(dist_x)
    # reconstruct responses
    if args.recon == 'rbf':
        pred = recon_rbf(dist_array.T, target, anchors_idx_y, args.rbf_gamma).flatten()
    elif args.recon == 'optimize':
        pred = recon_optimize(np.maximum(0, dist_array), target.loc[anchors_idx_y].values.flatten(), args.opt_method)
    elapsed_time = time.time() - start_time
    return pred, elapsed_time


def recon_optimize(dist_array, anchor_response, method):
    """
    Reconstructs response values from response distance estimations using optimization.

    Parameters:
    - dist_array (np.ndarray): A 2D array stores response distance estimations between samples and response anchors. Each row represents the distances from a 
      sample to the response anchors.
    - anchor_response (np.ndarray): A 1D array of response values corresponding to the response anchors.
    - method (str): The optimization method used in `scipy.optimize.minimize` (e.g., 'Nelder-Mead').

    Returns:
    - np.ndarray: A 1D array of predicted response values for the test samples.
    """
    def estimate_response(distances, y_anchors):
        def objective(y_test):
            return np.sum((distances - np.abs(y_test - y_anchors))**2)
        
        # Initial guess (mean of the anchor responses)
        y_test_init = np.mean(y_anchors)
        
        # Optimization
        result = minimize(objective, y_test_init, method=method)
        return result.x[0]  # Return the estimated response value

    # Loop over all test samples
    pred = np.array([estimate_response(distances, anchor_response) for distances in dist_array])
    return pred.flatten()


def recon_rbf(dist_array, target, anchors_idx, gamma=1) -> np.array:
    """
    Reconstructs response values from response distance estimations using RBF-based weights.

    Parameters:
    - dist_array (np.ndarray): A 2D array stores response distance estimations between samples and response anchors. Each column represents the distances from a 
      sample to the response anchors.
    - target (pd.DataFrame): Target values of the response anchors
    - anchors_idx (list): Indices of reponse anchors
    - gamma (float): gamma of RBF.

    Returns:
    - np.ndarray: A 1D array of predicted response values for the test samples.
    """
    def _rbf(x, gamma):
        return np.exp(-(x/gamma)**2)
    #
    target_anchor = target.loc[anchors_idx]
    if target_anchor.ndim == 1:
        target_anchor = target_anchor.values.reshape(-1, 1)
    #
    rbf_v = np.vectorize(_rbf)
    k = rbf_v(dist_array, gamma).T  # rbf of distance. n_t x n_a
    # print(k.sum(axis=1))
    h = np.linalg.inv(np.diag(k.sum(axis=1)))  # normalize mat. n_test x n_test
    r = np.asarray(target_anchor)# .values  # real y. n_anchors x n_features.
    rt = h @ k @ r  # np.matmul. Does it work?
    
    return rt


def anchor_select_pred_train_err(target, train_idx, pred_train, num_anchors):
    """
    Selects training samples with the largest prediction errors as structure anchors.

    Parameters:
    - target ( pd.DataFrame): Target values of training samples.
    - train_idx (list): Indices of training samples.
    - pred_train (np.ndarray): Predicted target values for training samples.
    - num_anchors (int): Number of structure anchors to select.

    Returns:
    - list: Indices of the training samples with the largest prediction errors.
    """
    # calculate the differences between the dist_y_train_est and dist_y_train
    err_train = abs(target.loc[train_idx].squeeze() - pred_train.flatten())
    # Get the index of the samples with the largest err
    anchors_idx = err_train.nlargest(num_anchors).index.tolist()
    return anchors_idx