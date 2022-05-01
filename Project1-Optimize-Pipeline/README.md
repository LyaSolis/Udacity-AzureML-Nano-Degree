# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

Bank Marketing Data Set contains records about call marketing campaigns of a Portuguese bank offering product (bank term deposit) subscription. 
Often multiple calls were required in order to establish if a client would subscribe or not.
The goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).[[source]](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

The best performing model was VotingEnsemble from AutoML pipeline (accuracy 0.918%).

## Scikit-learn Pipeline

Scikit-learn Pipeline is built using the Python SDK in Azure ML. Scikit-learn model was provided as part of prerequisites.

Scikit-learn Pipeline Steps:
 - Configure workspace and create an Experiment to hold our work
 - Provision a ComputeTarget - 'STANDARD_DS11_V2' VM with max 4 low priority nodes.
 - Create a train.py script, add: hyperparameter arguments (Logistic Regression model, learning_rate and batch_size), data load, data clean up and train/test split, model save steps.
 - Define a runtime Environment for training and specify:
   1. parameter sampler (RandomParameterSampling - supports discrete hyperparameters, early termination of low-performance runs. It's quicker and cheaper.) 
   2. early termination policy (BanditPolicy - starting at evaluation interval 5. Any run whose best metric is less than (1/(1+0.1) or 91% of the best performing run will be terminated.)
   3. estimator for the train.py script
   4. HyperDriveConfig using the estimator, hyperparameter sampler, and policy

 - Submit hyperparameter tuning run
 - Identify the best performing configuration and hyperparameter values and save them

## AutoML
AutoML run is configured to use our cleaned data as input to automatically train and tune multiple ML models with various hyperparameters, producing a large number of finetuned ML models as output. 
Models are automatically evaluated, which allows us to pick the best model at the end of the run.
Azure AutoML run steps:
 - Create TabularDataset using TabularDatasetFactory
 - Clean data and upload clean data to datastore (blob)
 - Set parameters for AutoMLConfig 
 - Run AutoML experiment
 - Retrieve and save best automl model (VotingEnsemble)

Parameters generated for AutoML were as follows:

<img width="525" alt="Screenshot 2022-04-06 at 15 17 01" src="https://user-images.githubusercontent.com/24227297/161996163-50dc11c8-3757-4174-880f-36f910c83692.png">
<img width="525" alt="Screenshot 2022-04-06 at 15 17 17" src="https://user-images.githubusercontent.com/24227297/161996386-17c81bf2-bc90-4a84-885c-7c70d771c3d5.png">

Where:
 - Primary metric was "accuracy" - ratio of predictions that exactly match the true class labels.
 - Early Stopping was enabled - this parameter is used to stop initation early if the score isn't improving.
 - Featurization was "auto" - this is to automatically check data and flag any issues (such as class imbalance, missing values etc).
 - Validation was set to 5 fold cross validation - 5 different trainings were performed, using 4/5 of data each time and 1/5 was held out each time.
 - Itiration time out was set to 10 min for each itiration.
 - Experiment time out was set to 30min, which means that it would terminate after 30 min, and best model is selected from the models generated within this time frame. This consists of all itirations within experiment.


VotingEnsemble model consists of several classifiers, each contributing a certain weight towards class prediction decision.

## Pipeline comparison
Scikit-learn Logistic Regression model has 0.907% accuracy.
AutoML generated VotingEnsemble -  0.918%.
This difference in performance can be attributed to automatic featurization weighting mechanism that AutoML automatically applies to imbalanced datasets.


## Future work

We had automatic featurization enabled in our AutoML run, which is a sequence of checks over the input data. This raised an alert about imbalanced classes detection in our dataset.
The 'accuracy paradox' is the case where accuracy measures are excellent (90%+) but it only reflects the underlying class distribution.
Therefore, for future work we could implement the following methods (for both, Scikit-Learn and AutoML):
 - change primary metric to F1, Kappa or ROC Curves
 - resample data (down-sample 'yes', up-sample 'no')
 - generate synthetic samples or use augmentation techniques
 
For Scikit-Learn:
 - optimize kernel for Logistic Regression model and given dataset

For AutoML:
 - optimize number of folds for cross-validation 

## Delete Compute Cluster

<img width="1543" alt="Screenshot 2022-04-06 at 14 21 57" src="https://user-images.githubusercontent.com/24227297/161987225-13d91653-49e9-43e9-ac21-a5628cee264f.png">

<img width="1543" alt="Screenshot 2022-04-06 at 14 22 23" src="https://user-images.githubusercontent.com/24227297/161987349-70df76e1-534e-4e78-bb54-83d92335941e.png">
