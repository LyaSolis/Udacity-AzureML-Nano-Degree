# Operationalizing Machine Learning

-------------------------------------------------
### Project Overview

The primary goal of a Machine Learning engineer is to deploy a trained model into production so that it can be consumed by others. These models must have a baseline, which is used to adapt and update the model when needed, and to constantly evaluate it in order to detect potential issues.  
Azure ML pipeline is an independently executable workflow of a complete machine learning task. Subtasks are encapsulated as a series of steps within the pipeline. 
In this project we will learn to configure production environments in Azure ML Studio and the Python SDK to get robust deployments. We will complete steps of setting up the right type of authentication, training a binary classification model using AutoML, deploying the best model, and then automating this task by creating and publishing an Azure Machine Learning pipeline with endpoint available for consumption.  

The key advantages of using pipelines for machine learning workflows are:

  * Pipelines allow you to focus on other work while lenghty tasks are running unattended. These tasks can be scheduled to run in parallel or in sequence, time-consuming steps can be done only when their input changes, independent steps allow collaboration of multiple people on the same pipeline.  
  * Individual pipeline steps can be run on different compute targets, such as HDInsight, GPU Data Science VMs, and Databricks.  
  * Reusability	by create pipeline templates (retraining, batch-scoring etc) and trigger published pipelines from external systems via simple REST calls.  
  * Tracking and versioning	is done automatically.  
  * Azure Machine Learning automatically orchestrates all of the dependencies between pipeline steps. This orchestration might include spinning up and down Docker images, attaching and detaching compute resources, and moving data between the steps in a consistent and automatic manner.

------------------------------------------------
### Architectural Diagram Of The Project

<img width="960" alt="Screenshot 2022-04-26 at 08 37 02" src="https://user-images.githubusercontent.com/24227297/165247188-65f723d9-c012-4545-97b2-3466fd0eb338.png">

------------------------------------------------
### Key Steps
##### In this project we will complete the below steps in order to create, publish and run a Pipeline object:

  * Authentication
  * Automated ML Experiment
  * Deploy the best model
  * Enable logging
  * Swagger Documentation
  * Consume model endpoints
  * Create and publish a pipeline
  * Documentation

##### Lets have a closer look at each of these stes:

1. #### Enable Security and Authentication (Login to ensure that the account is authentificated; Ensure az command-line tool is installed along with the ml extension; create a Service Principal; Capture the "objectId" using the clientID; Give the Service Principal acc the role of owner for the given Workspace, Resource Group and User objectId).  
   A “Service Principal” is a user role with controlled permissions to access specific resources. Creating a SP enables Continuous Delivery platform, like   ML Pipeline to train models. SP enhances security by allowing authentication while reducing the scope of permissions.  
   Ensure the az command-line tool is installed along with the ml extension (in order to be able to interact with Azure Machine Learning Studio). Then       create the Service Principal with az, capture the "objectId" using the clientID and assign the role to the new Service Principal for the given             Workspace, Resource Group and User objectId.  
   
   <img width="1098" alt="Screenshot 2022-04-18 at 16 00 28" src="https://user-images.githubusercontent.com/24227297/163827677-54f625c7-8f45-48a4-9ff3-90d7765f371a.png">  
   
   To enable Security and Authentication, log in with az, capture the "objectId" using the clientID (this was generated during previous step when we         created SP), and finally run command `az ml workspace share` to allow the Service Principal access to the workspace.  
   
   <img width="1098" alt="Screenshot 2022-04-18 at 16 04 27" src="https://user-images.githubusercontent.com/24227297/163828254-68248930-98b8-47e2-861d-025d8062c769.png">  
   
2. #### Create AutoML run to train a binary classification model. This task also requires uloading dataset, creating and configuring compute, specifying training task parameters: task 'classification', target column, evaluation metric (AUC), validation method k-fold cross-val (5 folds).  

    * Register BankMarketing dataset in ML Studio. URL was used to download and register BankMarketing dataset as AML Dataset in Workspace.  
  
    <img width="1284" alt="Screenshot 2022-04-18 at 16 26 32" src="https://user-images.githubusercontent.com/24227297/163831462-385f9170-e5c6-499d-973f-9e328d66007f.png">  
  
    * AutoML experiment is completed, a range of classification models is trained, the best model is going to be selected for deployment.   
  
    <img width="1293" alt="Screenshot 2022-04-19 at 07 26 16" src="https://user-images.githubusercontent.com/24227297/163939527-766b2b5b-6975-49a3-a583-54a04dd3efa6.png">  
  
3. #### Deploy the best ML model from AutoML run.  
   Model Deployment has two parts: first we need to **Configure Deployment Settings**, and then **Deploy Model**. 
     * Go to Automated ML section and find our recent experiment with a completed status
     * Go to the "Model" tab and select the best model generated during the AutoML experiment (VotingEnsemble `AUC weighted 0.95036`)  
   
   <img width="933" alt="Screenshot 2022-04-24 at 11 14 04" src="https://user-images.githubusercontent.com/24227297/164971637-1f5947a2-6ccf-4839-88e7-9f6e27af2393.png">  
   
     * Click "Deploy" button above it
     * Fill out the form with a name and description
     * Select Azure Container Instance (ACI) for Compute Type
     * Enable Key-based Authentication - by default it's disabled in ACI service (Token-based auth isn't supported in ACI)

4. #### Enable Application Insights after a model is deployed 
   Application Insights is an Azure service which provides key facts about an application, detects anomalies and visualizes performance of the deployment.    It can be enabled before or after a deployment.  
   Download workspace config.json and put it in the same directory with the rest of the files used in this project. Run               `service.update(enable_app_insights=True)` command to enable insights.
   
   <img width="655" alt="Screenshot 2022-04-19 at 07 33 51" src="https://user-images.githubusercontent.com/24227297/163940580-7f1f0503-ed96-4942-bdeb-c799fbb7a75d.png">  
   
5. #### Run logging and review details
   Logging - is an information output, usually in the form of text, produced by the software. Logs output is used to debug problems in deployed              containers.  
   Update the `logs.py` with information from the Endpoints section. This file allows dynamic Azure authentification, App Insight enablement and logs        display. Showing below is an extract of what you should see in a successful response to a scoring request.  
   
   <img width="1094" alt="Screenshot 2022-04-19 at 07 35 04" src="https://user-images.githubusercontent.com/24227297/163940761-df0210df-2eeb-4db4-93cd-3e9d92a123db.png">  
   
   When HTTP requests are submitted to a deployed model, there are three HTTP error codes that might come be returned in logs. Here is how to troubleshoot    them:  
     * HTTP STATUS 502: After a deployment, the application crashes because of an unhandled exception - debug code.
     * HTTP STATUS 503: When there are large spikes in requests, the system may not be able to cope with all of them and some clients may see this code - wait and try again.
     * HTTP STATUS 504: The request timed out. In Azure, the requests time out after 1 minute. If the score.py script is taking longer than a minute, this error code will be produced - get more powerful compute cluster.  

6. #### Swagger Documentation  
   Swagger is a tool that helps build, document, and consume RESTful web services and explains what types of HTTP requests that API can consume (like POST    and GET). We downloaded and saved swagger.json provided by Azure - this json file is used to create a web site that documents the HTTP endpoint for a      deployed model.  
     * Run `swagger.sh` to start docker
     * Run `serve.py` - this will enable CORS so that swagger can build documentation locally
   Swagget UI shows what are the avilable endpoints and it has demo inputs for endpoints  
   
   <img width="973" alt="Screenshot 2022-04-24 at 07 49 40" src="https://user-images.githubusercontent.com/24227297/164960583-1f368b48-bdac-4373-a0c2-7bc4c82317ba.png">  
   
7. #### Consume Deployed Service  
   A deployed service can be consumed via an HTTP API - which is an exposed over the network URL. Users can interact with the trained model by initiating    HTTP requests using JSON (JavaScript Object Notation) to accept data and submit responses. JSON is a bridge language among different environments..        HTTP **POST** request method is used to submit data. HTTP **GET** is used to retrieve information from a URL.  
     * Update `endpoint.py` with full URL to the endpoint and the key to authenticate
     * Run `python endpoint.py` to interact with the endpoint  
   We made an authenticated HTTP request to a deployed model service in Azure Container Services to retrieve output from the model:  
   
   <img width="771" alt="Screenshot 2022-04-24 at 08 05 50" src="https://user-images.githubusercontent.com/24227297/164962140-3cfd8e2e-6b71-4038-bfb7-173c94007463.png">  

8. #### Benchmark the Endpoint
   Apache Benchmark (ab) is an easy and popular tool for benchmarking HTTP services. It runs against the HTTP API using authentification key and score URL    to retrieve performance results. One of the most significant performance metrics for a deployed model is the average response time, because Azure times    out if the response times are longer than 60 seconds.  
   We can either run `benchmark.sh` script or `ab -n 10 -v 4 -p data.json -T 'application/json' -H 'Authorization: Bearer SECRET' http://URL.azurecontainer.io/score` command. In either case authentification key and score URL must be udated according to the deployment details. Ab        command runs against the selected endpoint using the data.json file created by `endpoint.py` file in the previous step.
   Command ab (Apache Benchmark) arguments:  
     * `-n 10` 10 requests  
     * `-v 4` increased verbosity output  
     * `-p data.json` posts data.json file generated in previous step by `endpoint.py`  
     * `-T 'application/json'` send it as json application  
     * `-H 'Authorization` key and url used in endpoint.py file  

   <img width="771" alt="Screenshot 2022-04-24 at 08 21 12" src="https://user-images.githubusercontent.com/24227297/164964942-f4ee0b97-ee77-431d-b739-e9bc90399e10.png">  

   The output shows us that `Time per request: 276.377 [ms] (mean)` - which is super fast; no failed requests `Failed requests:  0`  
   
9. #### Workflow Automation
   A great way to automate workflows is via publishing Pipelines. Automation connects different services and actions together to be part of a new workflow    that wasn’t possible before.
   First we need to set up the workspace. This included importing required libraries, getting Workspace details from config, setting Experiment name and      folder and creating AmlCompute.  
   Pipelines can take configuration and different steps, which in turn can have different arguments and parameters (i.e. variables in a Python script). We    will create a pipeline with AutoML steps. Pipeline Parameters are also available as a class and can be configured with the various different parameters    needed for later use.    
   Running Jupyter Notebook, we have created our pipeline for training BankMarketing client binary classification model:  
   
   <img width="1575" alt="Screenshot 2022-04-24 at 09 20 57" src="https://user-images.githubusercontent.com/24227297/164967230-7623e0bb-36d8-404c-b7a3-876598294c46.png">  
   
   Pipeline Section of Azure ML Studio showing Pipeline Endpoint:

   <img width="1258" alt="Screenshot 2022-04-24 at 12 26 14" src="https://user-images.githubusercontent.com/24227297/164974143-7c1fa9e4-a13b-4498-b0fb-9f86f3976119.png">  

   The "Published Pipeline overview" showing a REST endpoint and a status of "ACTIVE":  

   <img width="1611" alt="Screenshot 2022-04-24 at 12 45 21" src="https://user-images.githubusercontent.com/24227297/164974830-205a3f61-66bd-474d-bac1-40b0055c3e1e.png">  

   In Notebook "Use RunDetails Widget" step runs:  

   <img width="1325" alt="Screenshot 2022-04-24 at 12 48 41" src="https://user-images.githubusercontent.com/24227297/164974943-3a6f4073-c894-43ad-816a-24584775d7fc.png">  

   In ML Studio showing scheduled run:
   
   To schedule a recurring Pipeline, you must pass the information necessary to set the interval using the ScheduleRecurrence class into the create() method of the Schedule class as a recurrence value.
   <img width="938" alt="Screenshot 2022-04-25 at 09 10 13" src="https://user-images.githubusercontent.com/24227297/165047351-268ddd2d-a1ef-4b2a-8814-d0505dcc8b51.png">

----------------------------------------------------------

## Screen Recording
https://youtu.be/gXg2FUkLWtw

------------------------------------------------
## Standout Suggestions

Our AutoML run had automatic featurization enabled. This feature carries out a sequence of checks over the input data, and, in our case it raised an alert about imbalanced classes detection.  

<img width="1011" alt="Screenshot 2022-04-24 at 16 30 42" src="https://user-images.githubusercontent.com/24227297/164984032-e900489f-2e94-495e-8f0e-f350b1405d48.png">   

Therefore, for future work we could implement the following methods in AutoML step:
  * Resample data (down-sample 'yes', up-sample 'no')
  * Generate synthetic samples or use augmentation techniques
  * Optimize number of folds for cross-validation

 

