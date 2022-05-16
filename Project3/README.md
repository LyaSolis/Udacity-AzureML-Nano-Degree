# Text Classification With AzureML


This project serves as a capstone project for my Udacity AzureML Nano Degree. 
We will train scikit-learn classification models using Azure Machine Learning automated hyperparameter tuning and Azure AutoML Python SDK. 

We will then select the better model, deploy it as a webservice, and interact with it by posting text queries and receiving predicted label for the text. 

<img width="753" alt="Screenshot 2022-05-16 at 18 05 58" src="https://user-images.githubusercontent.com/24227297/168646045-1c9abdfc-7ce6-44d8-8d1b-29ac02be3ee4.png">

## Dataset

### Overview

#### The 20 newsgroups text dataset[^1]
The [20 newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) comprises around 18000 newsgroups posts on 20 topics split in two subsets (train and test). 

Here is an example of text from the dataset:

<img width="727" alt="Screenshot 2022-05-16 at 10 55 52" src="https://user-images.githubusercontent.com/24227297/168568064-6a001c2c-23b8-47f5-b85f-ca243d41544e.png">

### Task

This NLP task aims to automatically classify text and assign it to one of the groups we selected for this project.

### Access

Scikit-learn module contains two loaders: 
  * sklearn.datasets.fetch_20newsgroups (returns a list of the raw texts that can be fed to text feature extractors such as sklearn.feature_extraction.text.CountVectorizer with custom parameters so as to extract feature vectors)
  * sklearn.datasets.fetch_20newsgroups_vectorized (returns ready-to-use features, i.e., it is not necessary to use a feature extractor) 

Nearly every group is distinguished by headers and footers ("NNTP-Posting-Host:", "Distribution:", sender affiliations with a university, names and e-mail addresses of particular people who were posting at the time etc), quotes (referring to previous posts, ex: “In article [article ID], [name] <[e-mail address]> wrote:”)

With so many clues distinguishing newsgroups, the classifiers get overfitted and barely have to identify topics from text at all.

In this project I will use sklearn.datasets.fetch_20newsgroups to load raw text, and limit it to only 4 categories in order to save processing power:  

```
categories = [
        "rec.sport.baseball",
        "rec.sport.hockey",
        "comp.graphics",
        "sci.space",
    ]
```
In addition, to remove this information leakage we will use a parameter called `remove` with the loader function, telling it what kinds of information to strip out of each file. 

`remove = ("headers", "footers", "quotes")`

[^1]:Source https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html


## Automated ML

AutoML run is configured to use our formatted data as input to automatically train and tune multiple ML models with various hyperparameters, producing a large number of finetuned ML models as output. 
Models are automatically evaluated, which allows us to pick the best model at the end of the run. 
Azure AutoML run steps:

  * Use sklearn.datasets.fetch_20newsgroups function to load dataset, add parameters to subset 4 groups and remove features causing information leakage
  * Create TabularDataset using Dataset.Tabular function and upload clean data to datastore (blob)
  * Set parameters for AutoMLConfig
  * Run AutoML experiment
  * Retrieve and save best automl model (StandardScalerWrapper LogisticRegression)

Parameters generated for AutoML were as follows:
```
automl_settings = {
    "iteration_timeout_minutes": 10,
    "primary_metric": "accuracy",
    "max_concurrent_iterations": num_nodes,
    "max_cores_per_iteration": -1,
    "enable_dnn": True,
    "featurization": "auto",
    "enable_early_stopping": True,
    "validation_size": 0.3,
    "verbosity": logging.INFO,
    "enable_voting_ensemble": False,
    "enable_stack_ensemble": False,
}

automl_config = AutoMLConfig(
    experiment_timeout_minutes=60,
    task="classification",
    debug_log="automl_errors.log",
    compute_target=compute_target,
    training_data=train_dataset,
    label_column_name=target_column_name,
    blocked_models=["LightGBM", "XGBoostClassifier"],
    **automl_settings,
)
``` 

  * Primary metric "accuracy" - ratio of predictions that exactly match the true class labels.
  * Early Stopping enabled - this parameter is used to stop initation early if the score isn't improving.
  * Featurization "auto" - this is to automatically check data and flag any issues (such as class imbalance, missing values etc).
  * Validation 0.3 - 30% of data was held out for validation.
  * Itiration time out was set to 10 min for each itiration.
  * Experiment time out was set to 60min, which means that it would terminate after 60 min, and best model is selected from the models generated within this time frame. This consists of all itirations within experiment.
  * Blocked_models parameter to exclude some models that can take a longer time to train on some text datasets. If we were to remove models from the blocked_models list, experiment_timeout_hours parameter value would need to be used and increased in order to allow sufficient time to improve the results.

### Results

```
run_preprocessor: StandardScalerWrapper,
run_algorithm: LogisticRegression,
Accuracy: 0.91198
```

`RunDetails` widget results:

<img width="912" alt="Screenshot 2022-05-16 at 10 16 47" src="https://user-images.githubusercontent.com/24227297/168560492-0752ba8b-2517-4699-815d-f410d0122e79.png">


<img width="912" alt="Screenshot 2022-05-16 at 10 17 19" src="https://user-images.githubusercontent.com/24227297/168560459-9544821d-376b-4e7c-a6e3-0a0f1074d067.png">

Screenshot of the best model trained with it's parameters:

<img width="1109" alt="Screenshot 2022-05-16 at 10 24 26" src="https://user-images.githubusercontent.com/24227297/168561854-f9000b92-63ad-480e-98e3-5ef37b3078b2.png">

Training algorithm:
```
{
    "class_name": "LogisticRegression",
    "module": "sklearn.linear_model",
    "param_args": [],
    "param_kwargs": {
        "C": 24.420530945486497,
        "class_weight": null,
        "multi_class": "multinomial",
        "penalty": "l2",
        "solver": "lbfgs"
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
```

## Hyperparameter Tuning

Scikit-learn Pipeline is built using the Python SDK in Azure ML. 

Scikit-learn Pipeline Steps:

  * Configure workspace and create an Experiment to hold our work
  * Provision a ComputeTarget - 'STANDARD_NC6' GPU VM with max 1 nodes (quota limit).
  * Create a train.py script, add: hyperparameter arguments (Logistic Regression model, learning_rate and batch_size), data load, format data and train/test split, model save steps.
  * Define a runtime Environment for training and specify:
    1. parameter sampler (RandomParameterSampling - supports discrete hyperparameters, early termination of low-performance runs. It's quicker and cheaper.)
    2. early termination policy (BanditPolicy - starting at evaluation interval 5. Any run whose best metric is less than (1/(1+0.1) or 91% of the best performing run will be terminated.)
    3. estimator for the train.py script
    4. HyperDriveConfig using the estimator, hyperparameter sampler, and policy
  * Submit hyperparameter tuning run
  * Identify the best performing configuration and hyperparameter values and save them


### Results

How could you have improved it?
```
run_algorithm: LogisticRegression,
Accuracy: 0.863
```

`RunDetails` widget output: 

<img width="900" alt="Screenshot 2022-05-16 at 10 34 50" src="https://user-images.githubusercontent.com/24227297/168563664-3aa18d08-248c-41e4-a16b-7194cf4afb15.png">

Screenshot of the best model trained with it's parameters:

<img width="1332" alt="Screenshot 2022-05-16 at 10 39 40" src="https://user-images.githubusercontent.com/24227297/168564957-f76147c6-cc99-4134-83c6-54f2e87f5700.png">

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
<img width="1129" alt="Screenshot 2022-05-16 at 10 28 51" src="https://user-images.githubusercontent.com/24227297/168562519-dfbd2e70-3687-4c5c-a10f-a7f9492bbf5b.png">

<img width="813" alt="Screenshot 2022-05-16 at 10 31 33" src="https://user-images.githubusercontent.com/24227297/168563085-ec2c5990-8052-48ad-81a9-6b5f83a1640f.png">

<img width="1129" alt="Screenshot 2022-05-16 at 10 30 54" src="https://user-images.githubusercontent.com/24227297/168563059-b0b5aa7b-7438-4399-a420-5ed3c11c0992.png">

## Screen Recording

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
