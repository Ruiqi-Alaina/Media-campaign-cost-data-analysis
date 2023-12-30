#import packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import VotingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import pickle
from google.cloud import bigquery 
from google.cloud import aiplatform
from google.cloud import storage
import hypertune
import argparse
import os
import sys
# local parameters
parser =  argparse.ArgumentParser()
parser.add_argument('--project_id',dest ='project_id',type=str)
parser.add_argument('--bq_project',dest ='bq_project', type=str)
parser.add_argument('--bq_dataset',dest ='bq_dataset', type=str)
parser.add_argument('--bq_table', dest ='bq_table',type=str)
parser.add_argument('--region', dest = 'region', type=str)
parser.add_argument('--experiment', dest='experiment',type=str)
parser.add_argument('--series', dest='series', type=str)
parser.add_argument('--experiment_name',dest='experiment_name',type=str)
parser.add_argument('--run_name', dest = 'run_name', type = str)
# xgb&lightgbm
parser.add_argument('--x_seed', dest = 'x_seed', type = int)
parser.add_argument('--l_seed', dest = 'l_seed', type = int)
parser.add_argument('--x_objective', dest='x_objective', type = str)
parser.add_argument('--l_objective', dest='l_objective', type = str)
parser.add_argument('--x_max_depth', dest='x_max_depth',type=int)
parser.add_argument('--l_max_depth', dest='l_max_depth',type=int)
#xgb&cat
parser.add_argument('--x_grow_policy', dest='x_grow_policy', type = str)
parser.add_argument('--c_grow_policy', dest='c_grow_policy', type = str)
#lightgbm & cat
parser.add_argument('--l_learning_rate', dest='l_learning_rate', type = float)
parser.add_argument('--c_learning_rate', dest='c_learning_rate', type = float)
# xgb 
parser.add_argument('--eval_metric', dest='eval_metric', type = str)
parser.add_argument('--tree_method', dest='tree_method', type = str)
parser.add_argument('--eta', dest="eta", type=float)
parser.add_argument('--alpha', dest='alpha', type=float)
#lighgbm
parser.add_argument('--metric', dest='metric', type = str)
parser.add_argument('--feature_fraction', dest='feature_fraction', type=float)
parser.add_argument('--bagging_fraction', dest='bagging_fraction', type=float)
parser.add_argument('--verbose', dest='verbose', type=int)
parser.add_argument('--num_leaves', dest='num_leaves', type = int)
parser.add_argument('--lambda_l1', dest='lambda_l1', type=float)
#catboost
parser.add_argument('--random_seed', dest = 'random_seed', type = int)
parser.add_argument('--loss_function', dest='loss_function', type=str)
parser.add_argument('--depth', dest='depth', type = int)
parser.add_argument('--iterations', dest='iterations',type=int)
parser.add_argument('--l2_leaf_reg', dest='l2_leaf_reg', type=float)
args = parser.parse_args()
# create clients
bq =  bigquery.Client(project=args.project_id)
aiplatform.init(project = args.project_id, location = args.region)
#vertex ai experiment 
if args.run_name in [run.name for run in aiplatform.ExperimentRun.list(experiment = args.experiment_name)]:
    expRun = aiplatform.ExperimentRun(run_name = args.run_name, experiment = args.experiment_name)
else:
    expRun = aiplatform.ExperimentRun.create(run_name = args.run_name, experiment = args.experiment_name)
expRun.log_params({'experiment':args.experiment, 'series':args.series, 'project_id': args.project_id})
expRun.log_params({'data_source':f"bq://{args.bq_project}.{args.bq_dataset}.{args.bq_table}"})
# extract data from bigquery table
# drop recyclable_packge and low_fat due to the feature importance we found in previous hyperparameter tuning job.
# feature engineering: creat home_children_ratio, drop prepared_food since high-correlation to salad_bar
query = f"SELECT store_sales_in_millions_, unit_sales_in_millions_, total_children-num_children_at_home AS independent_children,gross_weight, avg_cars_at_home_approx__1, units_per_case, store_sqft, coffee_bar+video_store+salad_bar+florist AS store_score, cost FROM `{args.bq_project}.{args.bq_dataset}.{args.bq_table}`"
df = bq.query(query).to_dataframe()
# extract target variable and explainatory variables
y = df['cost']
x = df.drop('cost', axis="columns")
# data preprocessing 
num_attrib = ["independent_children","avg_cars_at_home_approx__1", "store_sqft", "store_score"]
preprocess =  ColumnTransformer([("numerical", TargetEncoder(), num_attrib)],remainder='passthrough')
x_processed = preprocess.fit_transform(x,y)
# split data into train and test data sets
x_train,x_test,y_train,y_test =  train_test_split(x_processed, y, train_size = 0.8, test_size = 0.2, random_state=50)
#xgb_model parameters
xgb_params = {
    'seed': args.x_seed,
    'objective': args.x_objective,
    'eval_metric': args.eval_metric,
    'eta': args.eta,
    'max_depth': args.x_max_depth,
    'alpha': args.alpha,
    'tree_method': args.tree_method,
    'grow_policy': args.x_grow_policy
}
#lgbm_model parameters
lgbm_params = {
    'seed': args.l_seed,
    'objective': args.l_objective,
    'metric': args.metric,
    'learning_rate': args.l_learning_rate,
    'max_depth': args.l_max_depth,
    'lambda_l1': args.lambda_l1,
    'num_leaves': args.num_leaves,
    'bagging_fraction': args.bagging_fraction,
    'feature_fraction': args.feature_fraction,
    'verbose': args.verbose
}
#catboost_model parameters
catboost_params = {
    'random_seed': args.random_seed,
    'learning_rate': args.c_learning_rate,
    'iterations': args.iterations,
    'l2_leaf_reg': args.l2_leaf_reg,
    'depth': args.depth,
    'loss_function': args.loss_function,
    'grow_policy': args.c_grow_policy,
}
# build model 
xgb_model = XGBRegressor(**xgb_params)
lgbm_model = LGBMRegressor(**lgbm_params)
cat_model = CatBoostRegressor(**catboost_params)
ensemble_model = VotingRegressor([("xgb", xgb_model),("lgbm",lgbm_model),("catboost",cat_model)])
# train model 
model_ensemble = ensemble_model.fit(x_train, y_train)
# evaluate using msle metric
y_train_pred = model_ensemble.predict(x_train)
train_msle = mean_squared_log_error(y_train_pred, y_train)
y_test_pred =  model_ensemble.predict(x_test)
test_msle = mean_squared_log_error(y_test_pred, y_test)
expRun.log_params({"train_msle": train_msle, "test_msle": test_msle})
# save the model
file_name = 'model.pkl'
# Use predefined environment variable to establish model directionary
model_directory = os.environ['AIP_MODEL_DIR']
storage_path = f'/gcs/{model_directory[5:]}'+file_name
os.makedirs(os.path.dirname(storage_path), exist_ok=True)
# output the model save files directly to GCS destination
with open (storage_path,'wb') as f:
    pickle.dump(model_ensemble,f)
expRun.log_params({'model.save': storage_path})
expRun.end_run()
