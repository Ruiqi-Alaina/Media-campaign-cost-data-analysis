# Media-campaign-cost-data-analysis
Median campaign cost data from Kaggle gives information about the cost of media campaigns in the food marts in the United States on the basis of the features provided. The full description of the dataset could be found [here](https://www.kaggle.com/datasets/gauravduttakiit/media-campaign-cost-prediction).<br>
## Build with
SQL, tableau and python/R will be used for exploratory data analysis. Vertex AI on google cloud platform will be used for training custom model, deploying the model and making online prediction.
## Data
[Dataset for training and validating the model](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/train.csv)
## Roadmap
### Exploratory Data Analysis
* [SQL](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/SQL%20data%20exploration.sql)
* [Python](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/Explanatory%20data%20analysis.ipynb)
### Other data visualization in BI
[dashboard](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/dashboard.png)
### Hyperparameter tuning job 
Hyperparameter tuning job was implemented to find the optimized hyperparameters of xgboost, lightgbm and catboost model respectively. And considering the feature importance we got from these three models, some features were dropped when building the final voting model.
* [xgboost model hyperparameter tuning job on Vertex AI](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/xgboost.ipynb)
* [xgboost model training python script](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/xgbtrain.py)
* [lightgbm model hyperparameter tuning job on Vertex AI](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/LGBM.ipynb)
* [lightgbm model training python script](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/lgbmtrain.py)
* [catboost model hyperparameter tuning job on Vertex AI](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/catboost.ipynb)
* [catboost model training python script](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/cattrain.py)
### Ensemble model with voting regressor
Use the hyperparameter we found above to build model. Deploy the model and make onine prediction
* [Custom training job on Vertex AI](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/ensemble%20model.ipynb)
* [ensemble model with voting regressor training python script](https://github.com/Ruiqi-Alaina/Media-campaign-cost-data-analysis/blob/main/ensemble.py)
