*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project.
This project serves as a capstone project for my
NER and Relation Extraction from EHR. This repository includes code for NER and RE methods on EHR records. 

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
