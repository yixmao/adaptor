import numpy as np
import argparse
import sys
from adaptor.utils.utils import str2bool

def TRArgs(args=None):
    """
    Parses command-line arguments for adaptive Topological Regression (AdapToR).

    Parameters:
    - args (str, optional): Command-line argument string to parse (default: None). If None, uses sys.argv.

    Returns:
    - argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Adaptive Topological Regression (AdapToR) configuration.")

    # General settings
    parser.add_argument('-seed', type=int, default=10, help="Random seed for reproducibility.")
    parser.add_argument('-save_mdls', type=str2bool, default=False, help="Whether to save trained models.")
    parser.add_argument('-cpus', type=int, default=1, help="Number of CPUs used in distance calculation.")

    # Descriptor and distance configuration
    parser.add_argument('-distance', type=str, default='mhfp_jaccard', 
                        help="Structure distance metric used for computation: "
                             "'jaccard' (for ECFP4), 'mhfp_jaccard' (for HMFP6), "
                             "or other metrics supported by cdist.")
    
    # Response reconstruction
    parser.add_argument('-recon', type=str, default='optimize', help="Method for reconstructing responses from distance estimates (options: 'rbf', 'optimize').")
    parser.add_argument('-opt_method', type=str, default='Nelder-Mead', help="Optimization method supported by scipy.optimize.minimize.")
    parser.add_argument('-rbf_gamma', type=float, default=1, help="Gamma parameter for the RBF kernel.")

    # Model configurations
    parser.add_argument('-model', type=str, default='LR_L2', help="Regression model type ('LR' or 'LR_L2').")
    parser.add_argument('-ridge_alpha', type=float, default=0.05, help="Alpha (or lambda) parameter for ridge regression.")

    # TR configuration
    parser.add_argument('-anchor_x_sel', type=str, default='adaptive', 
                        help="Anchor selection strategy: "
                             "'adaptive' (adaptive anchor selection), 'random' (random selection).")
    parser.add_argument('-integrate_method', type=str, default=None, 
                        help="Method for integrating results: 'None', 'ensemble' or 'stack'.")
    parser.add_argument('-num_anchors_x', type=float, default=0.15, 
                        help="Number/percent or fraction of structure anchors. If >1: number, if <1: percent")
    parser.add_argument('-num_anchors_y', type=int, default=10, help="Number of response anchors.")
    parser.add_argument('-anchors_y_sel', type=str, default='cluster', 
                        help="Method for selecting response anchors: "
                             "'cluster' (K-means clustering) or 'same' (identical to structure anchors). "
                             "If 'same', overrides 'num_anchors_y'.")
    parser.add_argument('-num_steps', type=int, default=4, help="Number of steps/models to be trained.")
    parser.add_argument('-append_anchors', type=str2bool, default=True, 
                        help="Whether to accumulate anchor points across steps.")

    # Ensemble TR configuration
    parser.add_argument('-random_anchor_perc', type=str2bool, default=False, 
                        help="Whether to use a random percentage of structure anchors (overrides 'num_anchors_x' if 'True').")
    parser.add_argument('-mean_anchor_percentage', type=float, default=0.6, help="Mean percentage of structure anchors for Ensemble TR.")
    parser.add_argument('-std_anchor_percentage', type=float, default=0.2, help="Standard deviation of structure anchors percentage for Ensemble TR.")
    parser.add_argument('-min_anchor_percentage', type=float, default=0.3, help="Minimum percentage of structure anchors for Ensemble TR.")
    parser.add_argument('-max_anchor_percentage', type=float, default=0.9, help="Maximum percentage of structure anchors for Ensemble TR.")

    # Early stopping
    # Validation settings
    parser.add_argument('-val_frac', type=float, default=0, help="Fraction of training samples used for validation.")
    parser.add_argument('-early_stop', type=str2bool, default=False, help="Whether to enable early stopping.")
    parser.add_argument('-patience', type=int, default=1, help="Number of steps to wait before stopping due to lack of improvement.")
    parser.add_argument('-min_delta', type=float, default=0.005, 
                        help="Minimum change in NRMSE required to qualify as an improvement.")

    # Logging and verbosity
    parser.add_argument('-verbose', type=str2bool, default=False, help="Whether to print progress and intermediate metrics.")

    # Parse the arguments if provided
    if args is not None:
        sys.argv = [''] + args.split()
    parsed_args = parser.parse_args()

    # Ensure num_anchors_x is an integer if greater than 1
    if parsed_args.num_anchors_x > 1:
        parsed_args.num_anchors_x = int(parsed_args.num_anchors_x)

    return parsed_args
