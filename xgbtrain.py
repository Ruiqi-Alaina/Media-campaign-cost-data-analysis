from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
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
# import argument to local variables
parser = argparse.ArgumentParser()
parser.add_argument('--project_id',dest ='project_id',type=str)
parser.add_argument('--bq_project',dest ='bq_project', type=str)
parser.add_argument('--bq_dataset',dest ='bq_dataset', type=str)
parser.add_argument('--bq_table', dest ='bq_table',type=str)
parser.add_argument('--region', dest = 'region', type=str)
parser.add_argument('--experiment', dest='experiment',type=str)
parser.add_argument('--series', dest='series', type=str)
parser.add_argument('--experiment_name',dest='experiment_name',type=str)
parser.add_argument('--run_name', dest = 'run_name', type = str)
parser.add_argument('--seed', dest = 'seed', type = int)
parser.add_argument('--objective', dest='objective', type = str)
parser.add_argument('--eval_metric', dest='eval_metric', type = str)
parser.add_argument('--tree_method', dest='tree_method', type = str)
parser.add_argument('--grow_policy', dest='grow_policy', type = str)
#hyperparameters 
parser.add_argument('--eta', dest="learning_rate", type=float)
parser.add_argument('--max_depth', dest='max_depth',type=int)
parser.add_argument('--alpha', dest='reg_term', type=float)
args = parser.parse_args()
## creat clients 
bq = bigquery.Client(project = args.project_id)
aiplatform.init(project = args.project_id, location = args.region)
hpt = hypertune.HyperTune()
args.run_name = f"{args.run_name}-{hpt.trial_id}"
# vertex AI Experiment
if args.run_name in [run.name for run in aiplatform.ExperimentRun.list(experiment = args.experiment_name)]:
    expRun = aiplatform.ExperimentRun(run_name = args.run_name, experiment = args.experiment_name)
else:
    expRun = aiplatform.ExperimentRun.create(run_name = args.run_name, experiment = args.experiment_name)
expRun.log_params({'experiment':args.experiment, 'series':args.series, 'project_id': args.project_id})
# get schema from bigquery source
query =  f"SELECT * FROM `{args.bq_project}.{args.bq_dataset}.INFORMATION_SCHEMA.COLUMNS`WHERE TABLE_NAME='{args.bq_table}'"
schema = bq.query(query).to_dataframe()
expRun.log_params({'data_source':f"bq://{args.bq_project}.{args.bq_dataset}.{args.bq_table}"})
# get the data from bigquery
# feature engineering: creat home_children_ratio, drop prepared_food since high-correlation to salad_bar
query = f"SELECT store_sales_in_millions_, unit_sales_in_millions_, total_children-num_children_at_home AS independent_children,gross_weight, avg_cars_at_home_approx__1, recyclable_package, low_fat, units_per_case, store_sqft, coffee_bar+video_store+salad_bar+florist AS store_score, cost FROM `{args.bq_project}.{args.bq_dataset}.{args.bq_table}`"
df = bq.query(query).to_dataframe()
# extract target variable and explainatory variables
y = df['cost']
x = df.drop('cost', axis="columns")
# data preprocessing 
num_attrib = ["independent_children","avg_cars_at_home_approx__1", "store_sqft", "store_score"]
cat_attrib = ["recyclable_package", "low_fat"]
preprocess =  ColumnTransformer([("categorical",OneHotEncoder(), cat_attrib),("numerical", TargetEncoder(), num_attrib)],remainder='passthrough')
x_processed = preprocess.fit_transform(x,y)
# split data into train and test data sets
x_train,x_test,y_train,y_test =  train_test_split(x_processed, y, train_size = 0.8, test_size = 0.2, random_state=50)
# xgboost tree 
xgb_params = {
    'seed': args.seed,
    'objective': args.objective,
    'eval_metric': args.eval_metric,
    'eta': args.learning_rate,
    'max_depth': args.max_depth,
    'alpha': args.reg_term,
    'tree_method': args.tree_method,
    'grow_policy': args.grow_policy
}
xgb_model =  XGBRegressor(**xgb_params)
expRun.log_params({'eta':args.learning_rate, 'max_depth': args.max_depth, 'alpha': args.reg_term})
model_xgb = xgb_model.fit(x_train, y_train)
# test evaluations:
y_train_pred = model_xgb.predict(x_train)
test_msle = mean_squared_log_error(y_test, y_pred)
expRun.log_metrics({'test_msle': test_msle})
# training evaluations
y_pred_training = model_xgb.predict(x_train)
training_msle = mean_squared_log_error(y_train, y_pred_training)
expRun.log_metrics({'training_msle': training_msle})
# report hypertune info back to Vertex AI Training > Hyperparameter Tuning Job
hpt.report_hyperparameter_tuning_metric( hyperparameter_metric_tag =  'Mean_square_log_error', metric_value = test_msle)
file_name = 'xgb_model.pkl'
# Use predefined environment variable to establish model directionary
model_directory = os.environ['AIP_MODEL_DIR']
storage_path = f'/gcs/{model_directory[5:]}'+file_name
os.makedirs(os.path.dirname(storage_path), exist_ok=True)
# output the model save files directly to GCS destination
with open (storage_path,'wb') as f:
    pickle.dump(model_xgb,f)
expRun.log_params({'model.save': storage_path})
expRun.end_run()
