# Adaptive Topological Regression

## Introduction
This Python package provides implementations for Topological Regression (TR) [1] and Adaptive Topological Regression (AdapToR) [2]. AdapToR builds upon TR by incorporating three technical improvements and two novel approaches: adaptive anchor selection and optimization-based response reconstruction. As demonstrated in [2], AdapToR outperforms TR and deep learning models on the NCI60 GI50 dataset and 530 ChEMBL datasets, establishing itself as a powerful, robust, computationally efficient, and interpretable tool for QSAR modeling.

For more information, please refer to:
1. Zhang, Ruibo, et al. "Topological regression as an interpretable and efficient tool for quantitative structure-activity relationship modeling." Nature Communications 15.1 (2024): 5072. 
2. Mao, Yixiang, et al. "AdapToR: Adaptive Topological Regression for quantitative structure-activity relationship modeling." bioRxiv:https://www.biorxiv.org/content/10.1101/2025.04.02.646801v1

## Installation
**Step 1: Create a conda environment (optional)**
```bash
conda create --name adaptor python=3.8
conda activate adaptor
```

**Step 2: Install the extendtr package**

Clone the github repo:
```bash
git clone git@github.com:yixmao/adaptor.git
cd adaptor
```
Install the package
```bash
python setup.py sdist bdist_wheel
pip install .
```
If an error occurs indicating that there is no module named `setuptools` or `wheel`, you can install them by `pip install setuptools wheel`.

**Step 3: Verify installation**
```bash
python example_scripts/tr_examples.py
```
Expected output:
```
----------------- Running AdapToR with distances -----------------
Final test performance:
Spearman: 0.928117758522517
R2: 0.8595705045395112
RMSE: 0.52605834079766
NRMSE: 0.3747392366172627
PCC: 0.9344901726927825
Bias: 0.13047118399118787
----------------- Running TR with descriptors -----------------
Final test performance:
Spearman: 0.9162213979418459
R2: 0.8257112021988844
RMSE: 0.5860564907230982
NRMSE: 0.4174790986398188
PCC: 0.9166954265719479
Bias: 0.13243057687131635
----------------- Running Ensemble TR (enhanced) -----------------
Final test performance:
Spearman: 0.9240850939188995
R2: 0.8518272154882716
RMSE: 0.5403671808027721
NRMSE: 0.38493218170442506
PCC: 0.9314418786473777
Bias: 0.1479758305811111
----------------- Running AdapToR (ensemble) -----------------
Final test performance:
Spearman: 0.9162213979418459
R2: 0.8505120987038364
RMSE: 0.5427599146137806
NRMSE: 0.3866366528100557
PCC: 0.9314438467416803
Bias: 0.14162483434662534
----------------- Test completed! -----------------
```

## Example usage
Detailed example usage of different TR configurations can be found in `tr_examples.ipynb`. Here, we go through a simple example that runs AdapToR on the CHEMBL dataset 278.

**Step 1: Prepare the data**

To use TR functions, first we need to prepare the data, including the distances (or descriptors), targets, train and test indices. Note that the distances (or descriptors) and targets need to be ```pd.DataFrame```, and the indices need to be ```list```.
```python
import pandas as pd
from adaptor.TR.distance import calculate_distance_parallel
# load the data
path = './example_datasets/CHEMBL278/'
# load the descriptor
desc = pd.read_parquet(f'{path}/data_mhfp6.parquet', engine='fastparquet')
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
```
It is recommended to save the distances for later uses
```python
# make sure that the indices of desc and target match
output_file = f'{path}/data_mhfp6_dist_mhfp_jaccard.parquet'
distance_x.to_parquet(output_file, engine='fastparquet', compression='gzip')
```
Then, we define the indices for training, test samples.
```python
# load the train and test indices
fold = 0
train_file = f"{path}/train_fold_{fold}.csv"
test_file = f"{path}/test_fold_{fold}.csv"
train_idx = pd.read_csv(train_file)['Compound_ID'].tolist()
test_idx = pd.read_csv(test_file)['Compound_ID'].tolist()
# make sure that train and test indices are included in target.index
train_idx = [idx for idx in train_idx if idx in target.index]
test_idx = [idx for idx in test_idx if idx in target.index]

##### alternatively, you can randomly split train and test idx
# from sklearn.model_selection import train_test_split
# dataset_idx = target.index.tolist()
# train_idx, test_idx = train_test_split(dataset_idx, test_size=0.2, random_state=args.seed)

```

**Step 2: Model and predict**

Set the arguments and random seed. You can use ```args = TRArgs()``` to receive command-line input.
```python
from adaptor.utils.args import TRArgs
from adaptor.utils.utils import set_seed
# get the args
# to get the models: -save_mdls 1
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
```
Train the TR model(s) and get the predictions using ```TopoReg```.
```python
from adaptor.TR.topoReg import TopoReg
# train the AdapToR model and get the predictions
metrics, preds_test, preds_val, preds_val_stack, \
pred_test, anchors_idx_x_all, anchors_idx_y, models = TopoReg(
    distance_x, distance_y, target, 
    train_idx, test_idx, args
)
```
Expected output:
```
Final test performance:
Spearman: 0.9317471566657725
R2: 0.8692531426774518
RMSE: 0.5075985496236505
NRMSE: 0.3615893490170143
PCC: 0.9397836203757282
Bias: 0.12001183270567459
```
Alternatively, you can use the function ```TopoReg_desc``` that takes descriptors and targtes as inputs instead of distances.
```python
from adaptor.TR.topoReg import TopoReg_desc
# train the AdapToR model and get the predictions
metrics, preds_test, preds_val, preds_val_stack, \
pred_test, anchors_idx_x_all, anchors_idx_y, models = TopoReg_desc(
    desc, target, train_idx, test_idx, args
)
```
Optionally, you can save the results for future use:
```python
import os
from adaptor.utils.utils import save_results, load_results
# To save the results
output_suffix = f'CHEMBL278_fold_{fold}'
results_folder = f'./results/'
os.makedirs(results_folder, exist_ok=True)
indices = {
    "train_idx": train_idx,
    "test_idx": test_idx,
    "anchors_idx_x_all": anchors_idx_x_all,
    "anchors_idx_y": anchors_idx_y
}
save_results(
    results_folder, output_suffix, metrics, preds_test, 
    preds_val, preds_val_stack, indices, models, args)
# save the metrics to a csv file
output_file = f'{results_folder}/test_metrics_results_{output_suffix}.csv'
pd.DataFrame(metrics).to_csv(output_file, index=False)

# To load the results
metrics, preds_test, preds_val, \
preds_val_stack, indices, models = load_results(results_folder, output_suffix)
# unpack the indices
train_idx = indices["train_idx"]
test_idx = indices["test_idx"]
anchors_idx_x_all = indices["anchors_idx_x_all"]
anchors_idx_y = indices["anchors_idx_y"]
```

## Parameters
**General settings**
Table below shows the parameters and their possible values and descriptions.
| Parameter Name          | Possible Values/Range                                        | Default Value  | Description |
|-------------------------|-------------------------------------------------------------|---------------|-------------|
| `-seed`                | Integer                                                     | `10`          | Random seed for reproducibility. |
| `-save_mdls`           | `True` / `False`                                            | `False`       | Whether to save trained models. |
| `-cpus`                | Integer (`>=1`)                                             | `1`           | Number of CPUs used in distance calculation. |
| `-verbose`            | `True` / `False`                                            | `False`       | Whether to print progress and intermediate metrics. |

**TR configurations**
| Parameter Name          | Possible Values/Range                                        | Default Value  | Description |
|-------------------------|-------------------------------------------------------------|---------------|-------------|
| `-distance`            | `'jaccard'`, `'mhfp_jaccard'`, `'euclidean'`                | `'mhfp_jaccard'` | Structure distance metric used for computation. <br> Only used in `TopoReg_desc`.|
| `-model`               | `'LR'`, `'LR_L2'`                                           | `'LR_L2'`     | Regression model type. |
| `-ridge_alpha`         | Float (`>0`)                                               | `0.05`        | Alpha (or lambda) parameter for ridge regression. |
| `-anchor_x_sel`        | `'adaptive'`, `'random'`                                   | `'adaptive'`  | Anchor selection strategy. |
| `-num_anchors_x`       | Float (`<1` for fraction, `>1` for number)                 | `0.15`        | Number or fraction of structure anchors. |
| `-anchors_y_sel`       | `'cluster'`, `'same'`                                      | `'cluster'`   | Method for selecting response anchors. |
| `-num_anchors_y`       | Integer (`>=1`)                                            | `10`          | Number of response anchors. |
| `-num_steps`           | Integer (`>=1`)                                            | `4`           | Number of steps/models to be trained. |
| `-append_anchors`      | `True` / `False`                                            | `True`        | Whether to accumulate anchor points across steps. |
| `-recon`               | `'rbf'`, `'optimize'`                                       | `'optimize'`  | Method for reconstructing responses from distance estimates. |
| `-opt_method`          | String (valid `scipy.optimize.minimize` methods)           | `'Nelder-Mead'` | Optimization method used in response reconstruction. |
| `-rbf_gamma`           | Float (`>0`)                                               | `1`           | Gamma parameter for the RBF kernel. |
| `-integrate_method`    | `None`, `'ensemble'`, `'stack'`                            | `None`        | Method for integrating predictions across models. |

**Ensemble TR configurations**
| Parameter Name          | Possible Values/Range                                        | Default Value  | Description |
|-------------------------|-------------------------------------------------------------|---------------|-------------|
| `-random_anchor_perc`  | `True` / `False`                                            | `False`       | Whether to use a random percentage of structure anchors. |
| `-mean_anchor_percentage` | Float (`0-1`)                                          | `0.6`         | Mean percentage of structure anchors. |
| `-std_anchor_percentage`  | Float (`0-1`)                                          | `0.2`         | Standard deviation of structure anchors percentage. |
| `-min_anchor_percentage`  | Float (`0-1`)                                          | `0.3`         | Minimum percentage of structure anchors. |
| `-max_anchor_percentage`  | Float (`0-1`)                                          | `0.9`         | Maximum percentage of structure anchors. |

**Early Stop settings**
| Parameter Name          | Possible Values/Range                                        | Default Value  | Description |
|-------------------------|-------------------------------------------------------------|---------------|-------------|
| `-val_frac`            | Float (`0-1`)                                              | `0`           | Fraction of training samples used for validation. |
| `-early_stop`          | `True` / `False`                                            | `False`       | Whether to enable early stopping. |
| `-patience`           | Integer (`>=1`)                                            | `1`           | Number of steps to wait before stopping due to lack of improvement. |
| `-min_delta`          | Float (`>0`)                                               | `0.005`       | Minimum change in NRMSE required to qualify as an improvement. |


### Contact
If you have any questions or suggestions, please feel free to contact: Yixiang Mao (yixmao@ttu.edu) and Dr. Ranadip Pal (ranadip.pal@ttu.edu).
