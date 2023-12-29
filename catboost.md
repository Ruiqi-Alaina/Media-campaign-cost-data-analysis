# Hyperparameter tuning job with custom container


```python
# install google service usage/ artifact registry/ cloud build
try:
    import google.cloud.service_usage_v1
except ImportError:
    print('You need to pip install google-cloud-service-usage')
    ! pip install google-cloud-service-usage -q
try:
    import google.cloud.artifactregistry_v1 
except ImportError:
    print('You need to pip install google-cloud-artifact-registry')
    ! pip install google-cloud-artifact-registry -q
try:
    import google.cloud.devtools.cloudbuild
except ImportError:
    print("You need to pip install google-cloud-build")
    !pip install google-cloud-build
```

## Environment  set up


```python
project = !gcloud config get-value project
PROJECT_ID = project[0]
PROJECT_ID
```




    'my-project-media-campaign-cost'




```python
REGION = 'australia-southeast1'
EXPERIMENT = '03'
SERIES = '03'

# source data
BQ_PROJECT = PROJECT_ID
BQ_DATASET = 'media_campaign_cost'
BQ_TABLE = 'mcc_train'

# Resources
BASE_IMAGE = 'us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu113.py310'
DEPLOY_IMAGE = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest'
TRAIN_IMAGE = 'us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest'
TRAIN_COMPUTE = 'n1-standard-4'
DEPLOY_COMPUTE = 'n1-standard-4'

# Model Training

```


```python
# import packages
from google.cloud import aiplatform
from datetime import datetime
#import pkg_sourses
from IPython.display import Markdown as md
from google.cloud import service_usage_v1
from google.cloud.devtools import cloudbuild_v1
from google.cloud import artifactregistry_v1
from google.cloud import storage
from google.cloud import bigquery
from google.protobuf.struct_pb2 import Value
import json
import numpy as np
import pandas as pd

```


```python
# creat clients
aiplatform.init(project=PROJECT_ID, location=REGION)
bq = bigquery.Client(project=PROJECT_ID)
gcs = storage.Client(project=PROJECT_ID)
su_client = service_usage_v1.ServiceUsageClient()
ar_client = artifactregistry_v1.ArtifactRegistryClient()
cb_client = cloudbuild_v1.CloudBuildClient()
```


```python
# parameters
TIMESTAMP =datetime.now().strftime("%Y%m%d%H%M%S")
BUCKET = PROJECT_ID
URI = f"gs://{BUCKET}/{SERIES}/{EXPERIMENT}"
DIR = f"temp/{EXPERIMENT}"
```


```python
URI
```




    'gs://my-project-media-campaign-cost/03/03'




```python
SERVICE_ACCOUNT = !gcloud config list --format='value(core.account)'
SERVICE_ACCOUNT = SERVICE_ACCOUNT[0]
SERVICE_ACCOUNT 
```




    '683212519680-compute@developer.gserviceaccount.com'




```python
!rm -rf {DIR}
!mkdir -p {DIR}
```


```python
# Experiment Tracking
FRAMEWORK = 'sklearn'
TASK = 'regression'
MODEL_TYPE = 'catboost'
EXPERIMENT_NAME = f'experiment-{SERIES}-{EXPERIMENT}-{FRAMEWORK}-{TASK}-{MODEL_TYPE}'
RUN_NAME = f'run-{TIMESTAMP}'
```

## Get Vertex AI Experiment Tensorboard Instance Name


```python
tb = aiplatform.Tensorboard.list(filter = f'labels.series={SERIES}')
if tb:
    tb = tb[0]
else:
    tb = aiplatform.Tensorboard.create(display_name=SERIES, labels = {'series':f'{SERIES}'})
```

    Creating Tensorboard
    Create Tensorboard backing LRO: projects/683212519680/locations/australia-southeast1/tensorboards/5666091281185505280/operations/5787144212868759552
    Tensorboard created. Resource name: projects/683212519680/locations/australia-southeast1/tensorboards/5666091281185505280
    To use this Tensorboard in another session:
    tb = aiplatform.Tensorboard('projects/683212519680/locations/australia-southeast1/tensorboards/5666091281185505280')



```python
tb.resource_name
```




    'projects/683212519680/locations/australia-southeast1/tensorboards/5666091281185505280'



## Setup Vertex AI Experiments


```python
aiplatform.init(experiment = EXPERIMENT_NAME, experiment_tensorboard = tb.resource_name)
```

## Training 


```python
script_path = './trainer/cattrain.py'
with open(script_path, 'r') as file:
    data = file.read()
md(f"```python\n\n{data}\n```")
```




```python

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
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
parser.add_argument('--random_seed', dest = 'random_seed', type = int)
parser.add_argument('--loss_function', dest='loss_function', type=str)
parser.add_argument('--grow_policy', dest='grow_policy', type=str)
#hyperparameters
parser.add_argument('--learning_rate', dest='learning_rate', type = float)
parser.add_argument('--depth', dest='depth', type = int)
parser.add_argument('--iterations', dest='iterations',type=int)
parser.add_argument('--l2_leaf_reg', dest='reg_term', type=float)
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
#cat_attrib = ["recyclable_package", "low_fat"]
preprocess =  ColumnTransformer([("numerical", TargetEncoder(), num_attrib)],remainder='passthrough')
x_processed = preprocess.fit_transform(x,y)
# split data into train and test data sets
x_train,x_test,y_train,y_test =  train_test_split(x_processed, y, train_size = 0.8, test_size = 0.2, random_state=50)
# xgboost tree 
catboost_params = {
    'random_seed': args.random_seed,
    'learning_rate': args.learning_rate,
    'iterations': args.iterations,
    'l2_leaf_reg': args.reg_term,
    'depth': args.depth,
    'loss_function': args.loss_function,
    'grow_policy': args.grow_policy,
}
catboost_model =  CatBoostRegressor(**catboost_params)
expRun.log_params({'learning_rate':args.learning_rate, 'iterations': args.iterations, 'l2_leaf_reg': args.reg_term,'depth':args.depth})
model_catboost = catboost_model.fit(x_train, y_train)
# test evaluations:
y_pred = model_catboost.predict(x_test)
test_msle = mean_squared_log_error(y_test, y_pred)
expRun.log_metrics({'test_msle': test_msle})
# training evaluations
y_pred_training = model_catboost.predict(x_train)
training_msle = mean_squared_log_error(y_train, y_pred_training)
expRun.log_metrics({'training_msle': training_msle})
# report hypertune info back to Vertex AI Training > Hyperparameter Tuning Job
hpt.report_hyperparameter_tuning_metric( hyperparameter_metric_tag =  'Mean_square_log_error', metric_value = test_msle)
file_name = 'catboost_model.pkl'
# Use predefined environment variable to establish model directionary
model_directory = os.environ['AIP_MODEL_DIR']
storage_path = f'/gcs/{model_directory[5:]}'+file_name
os.makedirs(os.path.dirname(storage_path), exist_ok=True)
# output the model save files directly to GCS destination
with open (storage_path,'wb') as f:
    pickle.dump(model_catboost,f)
expRun.log_params({'model.save': storage_path})
expRun.end_run()

```



## Create a custom container with cloud buid


```python
#store resource in cloud storage
bucket = gcs.lookup_bucket(BUCKET)
if not bucket:
    gcs.bucket(BUCKET).create(location=REGION)
SOURCEPATH = f'{SERIES}/{EXPERIMENT}/training'

```


```python
# copy training code
blob = storage.Blob(f'{SOURCEPATH}/{EXPERIMENT}_trainer/train.py', bucket=gcs.bucket(BUCKET))
blob.upload_from_filename(script_path)
```


```python
# create requirements.txt file for python
requirements = f""" google-cloud-aiplatform
protobuf
db-dtypes>=1.0.0
google-auth>=2.6.0
google-cloud-bigquery>=3.0.1
cloudml-hypertune
catboost
"""
blob = storage.Blob(f'{SOURCEPATH}/requirements.txt', bucket=gcs.bucket(BUCKET))
blob.upload_from_string(requirements)
```


```python
# create the Dockerfile 
dockerfile = f'''
FROM {BASE_IMAGE}
WORKDIR /training
# copy requirements and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt
## Copies the trainer code to the docker image
COPY {EXPERIMENT}_trainer/* ./{EXPERIMENT}_trainer/
## Sets up the entry point to invoke the trainer
ENTRYPOINT ["python", "-m", "{EXPERIMENT}_trainer.train"]
'''


```


```python
blob = storage.Blob(f'{SOURCEPATH}/Dockerfile', bucket=gcs.bucket(BUCKET))
blob.upload_from_string(dockerfile)
```


```python
 for repo in ar_client.list_repositories(parent = f"projects/{PROJECT_ID}/locations/{REGION}"):
        print(repo.labels['experiment'])
```

    01
    02
    03



```python
## create docker image repository 
docker_repo = None
for repo in ar_client.list_repositories(parent = f"projects/{PROJECT_ID}/locations/{REGION}"):
    if repo.labels['experiment']==EXPERIMENT:
        docker_repo = repo
        print(f'Retrieved existing repo:{docker_repo.name}')
if not docker_repo:
    operation = ar_client.create_repository(
    request = artifactregistry_v1.CreateRepositoryRequest(
        parent = f'projects/{PROJECT_ID}/locations/{REGION}',
        repository_id =f'{PROJECT_ID}3',
        repository = artifactregistry_v1.Repository(
            description=f'A repository for the {EXPERIMENT} experiment',
            name = f'{PROJECT_ID}',
            format_ = artifactregistry_v1.Repository.Format.DOCKER,
            labels = {'series':SERIES, 'experiment':EXPERIMENT}
    )
    )
    )
    print ("Creating Repository ...")
    docker_repo = operation.result()
    print(f'Complete creating repo: {docker_repo.name}')

```

    Retrieved existing repo:projects/my-project-media-campaign-cost/locations/australia-southeast1/repositories/my-project-media-campaign-cost3



```python
docker_repo.name, docker_repo.format_.name
```




    ('projects/my-project-media-campaign-cost/locations/australia-southeast1/repositories/my-project-media-campaign-cost3',
     'DOCKER')




```python
REPOSITORY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{docker_repo.name.split('/')[-1]}"
```


```python
REPOSITORY
```




    'australia-southeast1-docker.pkg.dev/my-project-media-campaign-cost/my-project-media-campaign-cost3'




```python
SOURCEPATH
```




    '03/03/training'




```python
# setup the build config with empty list of steps - these will be added sequentially
build = cloudbuild_v1.Build(
    steps = []
)
# retrieve the source
build.steps.append(
    {
        'name': 'gcr.io/cloud-builders/gsutil',
        'args': ['cp', '-r', f'gs://{PROJECT_ID}/{SOURCEPATH}/*','/workspace']
    }
)
# docker build
build.steps.append(
    {
        'name': 'gcr.io/cloud-builders/docker',
        'args': ['build', '-t', f'{REPOSITORY}/{EXPERIMENT}_trainer','/workspace']
    }    
)

# docker push
build.images = [f"{REPOSITORY}/{EXPERIMENT}_trainer"]



operation = cb_client.create_build(
    project_id = PROJECT_ID,
    build = build
)

response = operation.result()

```


```python
response.status, response.artifacts
```




    (<Status.SUCCESS: 3>,
     images: "australia-southeast1-docker.pkg.dev/my-project-media-campaign-cost/my-project-media-campaign-cost3/03_trainer")



## Setup training job


```python
RANDOM_SEED = '50'
LOSS_FUNCTION = 'RMSE'
GROW_POLICY = 'Lossguide'
```


```python
CMDARGS = [
    "--project_id=" + PROJECT_ID,
    "--bq_project=" + BQ_PROJECT,
    "--bq_dataset=" + BQ_DATASET,
    "--bq_table=" + BQ_TABLE,
    "--region=" + REGION,
    "--experiment=" + EXPERIMENT,
    "--series=" + SERIES,
    "--experiment_name=" + EXPERIMENT_NAME,
    "--run_name=" + RUN_NAME,
    "--random_seed=" + RANDOM_SEED,
    "--loss_function=" + LOSS_FUNCTION,
    "--grow_policy="+GROW_POLICY
]

MACHINE_SPEC = {
    "machine_type": TRAIN_COMPUTE,
    "accelerator_count": 0
}

WORKER_POOL_SPEC = [
    {
        "replica_count":1,
        "machine_spec":MACHINE_SPEC,
        "container_spec":{
            "image_uri": f"{REPOSITORY}/{EXPERIMENT}_trainer",
            "args":CMDARGS,
        },
    }
]
```


```python
CMDARGS
```




    ['--project_id=my-project-media-campaign-cost',
     '--bq_project=my-project-media-campaign-cost',
     '--bq_dataset=media_campaign_cost',
     '--bq_table=mcc_train',
     '--region=australia-southeast1',
     '--experiment=03',
     '--series=03',
     '--experiment_name=experiment-03-03-sklearn-regression-catboost',
     '--run_name=run-20231228212543',
     '--random_seed=50',
     '--loss_function=RMSE',
     '--grow_policy=Lossguide']




```python
customJob = aiplatform.CustomJob(
    display_name = f'{SERIES}_{EXPERIMENT}_{TIMESTAMP}',
    worker_pool_specs = WORKER_POOL_SPEC,
    base_output_dir = f"{URI}/models/{TIMESTAMP}",
    staging_bucket = f"{URI}/models/{TIMESTAMP}",
    labels = {'series': f'{SERIES}', 'experiment':f'{EXPERIMENT}','experiment_name':f'{EXPERIMENT_NAME}'}
)
```

## Setup Hyperparameter Tuning Job


```python
METRIC_SPEC = {
    "Mean_square_log_error": "minimize"
}
PARAMETER_SPEC = {
    "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=1e-2, max=1, scale='linear'),
    "depth": aiplatform.hyperparameter_tuning.IntegerParameterSpec(min=4, max=10, scale='linear'), 
    "l2_leaf_reg":aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=0, max=2, scale="linear"),
    "iterations": aiplatform.hyperparameter_tuning.IntegerParameterSpec(min=100, max=1000, scale='linear')
}

```


```python
tuningJob = aiplatform.HyperparameterTuningJob(
    display_name = f'{SERIES}_{EXPERIMENT}_{TIMESTAMP}',
    custom_job = customJob,
    metric_spec = METRIC_SPEC,
    parameter_spec = PARAMETER_SPEC,
    max_trial_count = 18,
    parallel_trial_count =3,
    search_algorithm = None,
    labels = {'series':f'{SERIES}', 'experiment':f'{EXPERIMENT}', 'experiment_name':f'{EXPERIMENT_NAME}'}
)
```

## Run training job


```python
tuningJob.run(
    service_account = SERVICE_ACCOUNT
)
tuningJob.resource_name, tuningJob.display_name
```

    Creating HyperparameterTuningJob
    HyperparameterTuningJob created. Resource name: projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840
    To use this HyperparameterTuningJob in another session:
    hpt_job = aiplatform.HyperparameterTuningJob.get('projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840')
    View HyperparameterTuningJob:
    https://console.cloud.google.com/ai/platform/locations/australia-southeast1/training/6489416583180451840?project=683212519680
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_PENDING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_PENDING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_PENDING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_PENDING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_PENDING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_PENDING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_RUNNING
    HyperparameterTuningJob projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840 current state:
    JobState.JOB_STATE_SUCCEEDED
    HyperparameterTuningJob run completed. Resource name: projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840





    ('projects/683212519680/locations/australia-southeast1/hyperparameterTuningJobs/6489416583180451840',
     '03_03_20231228212543')




```python
job_link = f"https://console.cloud.google.com/ai/platform/locations/{REGION}/training/{tuningJob.resource_name.split('/')[-1]}?project={PROJECT_ID}"

print(f'Review the Job here:\n{job_link}')
```

    Review the Job here:
    https://console.cloud.google.com/ai/platform/locations/australia-southeast1/training/6489416583180451840?project=my-project-media-campaign-cost


## Get the best run


```python
mse  =  [trial.final_measurement.metrics[0].value if trial.state.name=='SUCCEEDED' else 1 for trial in tuningJob.trials]
best = tuningJob.trials[mse.index(min(mse))]
```


```python
tuningJob.trials
```




    [id: "1"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 7.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 550.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.505
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.0968000215489795
       }
     }
     start_time {
       seconds: 1703803519
       nanos: 763454680
     }
     end_time {
       seconds: 1703803866
     },
     id: "2"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 9.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 359.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.3926374943797586
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.29530480815871524
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09398148475233628
       }
     }
     start_time {
       seconds: 1703803519
       nanos: 763632467
     }
     end_time {
       seconds: 1703803872
     },
     id: "3"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 8.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 597.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 0.5526321997431658
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.7256656697311971
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.10079313382562005
       }
     }
     start_time {
       seconds: 1703803519
       nanos: 763689822
     }
     end_time {
       seconds: 1703803861
     },
     id: "4"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 9.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 147.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.8645769965027754
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.06268215178212838
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09439995871425683
       }
     }
     start_time {
       seconds: 1703804195
       nanos: 667662768
     }
     end_time {
       seconds: 1703804550
     },
     id: "5"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 9.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 136.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.888228625096389
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.5407097044869734
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09442877590389026
       }
     }
     start_time {
       seconds: 1703804197
       nanos: 190438130
     }
     end_time {
       seconds: 1703804563
     },
     id: "6"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 10.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 132.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 0.8801283081325364
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.03912615923989439
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.0946123423231848
       }
     }
     start_time {
       seconds: 1703804199
       nanos: 670520561
     }
     end_time {
       seconds: 1703804565
     },
     id: "7"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 10.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 392.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.5130664760843604
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.01
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09477526089801466
       }
     }
     start_time {
       seconds: 1703804892
       nanos: 743913282
     }
     end_time {
       seconds: 1703805234
     },
     id: "8"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 10.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 292.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.433471603145227
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.8218489956041096
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09819686288488164
       }
     }
     start_time {
       seconds: 1703804926
       nanos: 870660225
     }
     end_time {
       seconds: 1703805294
     },
     id: "9"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 6.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 293.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.551918990407277
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.5441280489376985
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.0953332514303597
       }
     }
     start_time {
       seconds: 1703804926
       nanos: 870794382
     }
     end_time {
       seconds: 1703805285
     },
     id: "10"
     state: INFEASIBLE
     parameters {
       parameter_id: "depth"
       value {
         number_value: 8.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 197.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 1.3146577035265552
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 1.0
       }
     }
     start_time {
       seconds: 1703805590
       nanos: 734648501
     }
     end_time {
       seconds: 1703805909
     },
     id: "11"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 10.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 355.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 0.7692261960502073
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.29396181646929465
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09407258237644396
       }
     }
     start_time {
       seconds: 1703805626
       nanos: 436393906
     }
     end_time {
       seconds: 1703805979
     },
     id: "12"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 10.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 100.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 2.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.30403279991806864
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09367732181651124
       }
     }
     start_time {
       seconds: 1703805661
       nanos: 578527557
     }
     end_time {
       seconds: 1703805992
     },
     id: "13"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 9.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 659.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 2.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.36837847434171556
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09532069124605363
       }
     }
     start_time {
       seconds: 1703806293
       nanos: 374946030
     }
     end_time {
       seconds: 1703806642
     },
     id: "14"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 9.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 100.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 2.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.2457074525900461
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09377747582516119
       }
     }
     start_time {
       seconds: 1703806293
       nanos: 377295259
     }
     end_time {
       seconds: 1703806606
     },
     id: "15"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 10.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 100.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 2.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.2650118572428292
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09378664459628214
       }
     }
     start_time {
       seconds: 1703806422
       nanos: 386946095
     }
     end_time {
       seconds: 1703806729
     },
     id: "16"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 4.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 100.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 2.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.23658426529087737
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09448617290576286
       }
     }
     start_time {
       seconds: 1703806987
       nanos: 966883104
     }
     end_time {
       seconds: 1703807303
     },
     id: "17"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 8.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 100.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 2.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.32841976672542367
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09370668371692664
       }
     }
     start_time {
       seconds: 1703806987
       nanos: 969253799
     }
     end_time {
       seconds: 1703807306
     },
     id: "18"
     state: SUCCEEDED
     parameters {
       parameter_id: "depth"
       value {
         number_value: 6.0
       }
     }
     parameters {
       parameter_id: "iterations"
       value {
         number_value: 100.0
       }
     }
     parameters {
       parameter_id: "l2_leaf_reg"
       value {
         number_value: 2.0
       }
     }
     parameters {
       parameter_id: "learning_rate"
       value {
         number_value: 0.31999925256622913
       }
     }
     final_measurement {
       step_count: 1
       metrics {
         metric_id: "Mean_square_log_error"
         value: 0.09377946917067802
       }
     }
     start_time {
       seconds: 1703807117
       nanos: 747648859
     }
     end_time {
       seconds: 1703807431
     }]




```python
best
```




    id: "12"
    state: SUCCEEDED
    parameters {
      parameter_id: "depth"
      value {
        number_value: 10.0
      }
    }
    parameters {
      parameter_id: "iterations"
      value {
        number_value: 100.0
      }
    }
    parameters {
      parameter_id: "l2_leaf_reg"
      value {
        number_value: 2.0
      }
    }
    parameters {
      parameter_id: "learning_rate"
      value {
        number_value: 0.30403279991806864
      }
    }
    final_measurement {
      step_count: 1
      metrics {
        metric_id: "Mean_square_log_error"
        value: 0.09367732181651124
      }
    }
    start_time {
      seconds: 1703805661
      nanos: 578527557
    }
    end_time {
      seconds: 1703805992
    }




```python
from tempfile import TemporaryFile
import pickle
storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET)
```


```python
pip install lightgbm
```

    Collecting lightgbm
      Downloading lightgbm-4.2.0-py3-none-manylinux_2_28_x86_64.whl.metadata (19 kB)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from lightgbm) (1.24.4)
    Requirement already satisfied: scipy in /opt/conda/lib/python3.10/site-packages (from lightgbm) (1.11.4)
    Downloading lightgbm-4.2.0-py3-none-manylinux_2_28_x86_64.whl (3.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m46.9 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hInstalling collected packages: lightgbm
    Successfully installed lightgbm-4.2.0
    Note: you may need to restart the kernel to use updated packages.



```python
blob = bucket.blob('03/03/models/20231228212543/12/model/catboost_model.pkl')
```


```python
blob
```




    <Blob: my-project-media-campaign-cost, 03/03/models/20231228212543/12/model/catboost_model.pkl, None>




```python
import catboost
with TemporaryFile() as temp_file:
    blob.download_to_file(temp_file)
    temp_file.seek(0)
    model=pickle.load(temp_file)
```


```python
model.feature_importances_
```




    array([10.1563797 ,  8.75559925, 67.0740398 ,  5.69041117,  2.75138447,
            0.98509972,  2.35870321,  0.29257844,  0.24004555,  1.6957587 ])




```python

```
