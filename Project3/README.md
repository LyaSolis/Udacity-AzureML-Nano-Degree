# Your Project Title Here

*TODO:* Write a short introduction to your project.
This project serves as a capstone project for my


Classification of text documents using sparse features

This is documentation for an old release of Scikit-learn (version 0.19). Try the latest stable release (version 1.1) or development (unstable) versions.

Classification of text documents using sparse features
This is an example showing how scikit-learn can be used to classify documents by topics using a bag-of-words approach. This example uses a scipy.sparse matrix to store the features and demonstrates various classifiers that can efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time (normalized) of each classifier.

NER and Relation Extraction from EHR. This repository includes code for NER and RE methods on EHR records. 
<img width="997" alt="Screenshot 2022-05-12 at 15 11 20" src="https://user-images.githubusercontent.com/24227297/168095265-2ddbcfc0-497c-4a1e-a1d5-fbb4be79d1b6.png">

## Project Set Up and Installation

*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview

[2018 Track 2: Adverse Drug Events and Medication Extraction in EHRs](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2018-t2/)  

The data consists of discharge summaries of nearly ~500 discharge summaries drawn from the MIMIC-III clinical care database, annotated by domain experts with entity tags and attributes were used to indicate the presence of drug and ADE information.

[Adverse Drug Events (ADE) Corpus](https://paperswithcode.com/dataset/ade-corpus)

A benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports.

### Task

This NLP task aims to automatically discover drug to adverse event (ADE) relations in clinical narratives and consists of three subtasks:

  * Concepts: Identifying drug names, dosages, durations and other entities.
  * Relations: Identifying relations of drugs with adverse drugs events (ADEs)[^1] and other entities given gold standard entities.
  * End-to-end: Identifying relations of drugs with ADEs and other entities on system predicted entities.

An adverse drug event (ADE) as "an injury resulting from medical intervention related to a drug".[^1]

As part of data preprocessing step, we augment **n2c2 2018** dataset to include a sample of **ADE corpus** dataset. 

[^1]:Definition by the World Health Organization (WHO).

### Access

Access to [n2c2 2018](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2018-t2/) is free, but requires registration. Once registration was completed (it took a couple of days to get approved), I downloaded the dataset locally and then registered it in my AzureML Datasets.

[Adverse Drug Events (ADE) Corpus](http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/) is available without registration. I downloaded these json files locally and then uploaded and registered them in my AzureML Datasets.

<img width="1688" alt="Screenshot 2022-05-13 at 08 06 58" src="https://user-images.githubusercontent.com/24227297/168256833-f916e954-a25f-4a06-8036-8dead979fdb4.png">

Notebook [generate_data.ipynb](https://github.com/LyaSolis/Udacity-AzureML-Nano-Degree/blob/main/Project3/generate_data.ipynb) shows how data was preprocessed for training.

<img width="1132" alt="Screenshot 2022-05-14 at 11 24 03" src="https://user-images.githubusercontent.com/24227297/168423191-7407569a-cd0e-44b9-9072-8fd79b2a2c96.png">

<img width="1132" alt="Screenshot 2022-05-14 at 11 24 03" src="https://user-images.githubusercontent.com/24227297/168423213-5e3856a8-bbf3-4a60-bc2b-6043270dc2a5.png">

Processed datasets were registered in datastore:

<img width="1437" alt="Screenshot 2022-05-14 at 11 58 02" src="https://user-images.githubusercontent.com/24227297/168423168-b4f4f2c6-d984-4fc2-8634-4aa91bff961d.png">


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
