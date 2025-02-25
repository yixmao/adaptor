"""
Test the installation of AdapToR package 
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from adaptor.utils.args import TRArgs
from adaptor.utils.utils import set_seed
from adaptor.TR.topoReg import TopoReg, TopoReg_desc
from adaptor.TR.distance import calculate_distance_parallel

if __name__ == "__main__":
    # load the data
    path = './example_datasets/CHEMBL278/'
    # load the descriptor
    desc = pd.read_parquet(f'{path}/data_mhfp6.parquet', engine='fastparquet')
    desc_ecfp4 = pd.read_parquet(f'{path}/data_ECFP4.parquet', engine='fastparquet').astype('bool')
    # load targets
    data = pd.read_csv(f'{path}/data_cp.csv', index_col=0)
    target = data["pChEMBL Value"].to_frame()
    # calculate the distances
    # structure distances
    distance_x = calculate_distance_parallel(desc, desc.index, desc.index, distance='mhfp_jaccard', cpus=1)
    distance_x = pd.DataFrame(distance_x, index=desc.index, columns=desc.index) 
    # response distances
    distance_y = pairwise_distances(target.values.reshape(-1, 1), metric="euclidean", n_jobs=-1)
    distance_y = pd.DataFrame(distance_y, index=target.index, columns=target.index)

    # load the train and test indices
    fold = 0
    train_file = f"{path}/train_fold_{fold}.csv"
    test_file = f"{path}/test_fold_{fold}.csv"
    train_idx = pd.read_csv(train_file)['Compound_ID'].tolist()
    test_idx = pd.read_csv(test_file)['Compound_ID'].tolist()
    # make sure that train and test indices are included in target.index
    train_idx = [idx for idx in train_idx if idx in target.index]
    test_idx = [idx for idx in test_idx if idx in target.index]


    print('----------------- Running AdapToR with distances -----------------')
   # get the args
    args = TRArgs("""
        -num_anchors_y 10 
        -anchors_y_sel cluster 
        -num_anchors_x 0.15 
        -anchor_x_sel adaptive 
        -num_steps 4 
        -model LR_L2 
        -recon optimize
    """)
    # set random seed
    set_seed(args.seed)
    # train the AdapToR model and get the predictions
    metrics, preds_test, preds_val, preds_val_stack, pred_test, anchors_idx_x_all, anchors_idx_y, models = TopoReg(distance_x, distance_y, target, train_idx, test_idx, args)

    print('----------------- Running TR with descriptors -----------------')
    # get the args
    args = TRArgs("""
        -distance jaccard
        -anchors_y_sel same 
        -num_anchors_x 0.6 
        -anchor_x_sel random 
        -num_steps 1 
        -model LR 
        -recon rbf
    """)
    # set random seed
    set_seed(args.seed)
    # train the TR model and get the predictions
    metrics, preds_test, preds_val, preds_val_stack, pred_test, anchors_idx_x_all, anchors_idx_y, models = TopoReg_desc(desc_ecfp4, target, train_idx, test_idx, args)

    print('----------------- Running Ensemble TR -----------------')
    # get the args
    args = TRArgs("""
        -distance jaccard
        -anchors_y_sel same
        -anchor_x_sel random
        -random_anchor_perc 1
        -num_steps 15 
        -model LR 
        -recon rbf
        -integrate_method ensemble
        -append_anchors 0
    """)
    # set random seed
    set_seed(args.seed)
    # train the AdapToR model and get the predictions
    metrics, preds_test, preds_val, preds_val_stack, pred_test, anchors_idx_x_all, anchors_idx_y, models = TopoReg_desc(desc_ecfp4, target, train_idx, test_idx, args)

    print('----------------- Running AdapToR (stack) -----------------')
    # get the args
    args = TRArgs("""
        -num_anchors_y 10 
        -anchors_y_sel cluster
        -num_anchors_x 0.15          
        -anchor_x_sel adaptive
        -num_steps 4
        -model LR_L2 
        -recon optimize
        -integrate_method stack
    """)
    # set random seed
    set_seed(args.seed)
    # train the AdapToR model and get the predictions
    metrics, preds_test, preds_val, preds_val_stack, pred_test, anchors_idx_x_all, anchors_idx_y, models = TopoReg(distance_x, distance_y, target, train_idx, test_idx, args)

    print('----------------- Test completed! -----------------')