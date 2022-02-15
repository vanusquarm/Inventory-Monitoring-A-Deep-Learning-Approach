# Inventory-Monitoring-A-Deep-Learning-Approach
Bin image classification based on the number of items present in the image.
This is an image classfication problem using an imagenet pretrained model called resnet-18 with sagemaker studio 
In the project, we profile the model's performance with respect to cpu, gpu, io and memory utilization. We first run the training with a higher hyperparameter ranges and then select the best hyperparameters to retrain our model. 

## Project Set Up and Installation
The project repository is cloned from the provided link to the udacity's github repo (deep-learning-topics-within-computer-vision-nlp-project-starter)

## Dataset
The dog breed dataset is used for the training 

### Access
The data is uploaded to the S3 bucket through the AWS Gateway so that SageMaker has access to the data, using sagemaker.Session().upload() api.
- s3://{bucket-name}/dogImages/

## Script Files used
1. `hpo.py` for hyperparameter tuning jobs where we train the model for multiple time with different hyperparameters and search for the best one based on loss metrics.
2. `train_model.py` for really training the model with the best parameters getting from the previous tuning jobs, and put debug and profiler hooks for debugging purpose.
3. `inference.py`: It includes the required methods (`model_fn` to load the model and `input_fn` to transform the input into something which can be understood by the model) for the model to be deployed.  


## Hyperparameter Tuning
I used a ResNet-18 pretrained model because it performs best for image classification tasks. Resnet is also resilient to the vanishing gradient problem, and the number of stacked layers does not degrade the network performance on the test dataset.

Below are hyperparameter types and their respective ranges used in the training
- learning rate 
- batch size
- epochs

```python
hyperparameter_ranges = {
    "batch-size": sagemaker.tuner.CategoricalParameter([32, 64, 128, 256, 512]),
    "lr": sagemaker.tuner.ContinuousParameter(0.01, 0.1),
    "epochs": sagemaker.tuner.IntegerParameter(2, 4)
}
```
The objective type is to maximize accuracy.

```python
objective_metric_name = "average test accuracy"
objective_type = "Maximize"
metric_definitions = [{"Name": "average test accuracy", "Regex": "Test set: Average accuracy: ([0-9\\.]+)"}]
```

Best hyperparameter values

```python
hyperparameters = {'batch-size': '512', 'lr': '0.026305482032806977', 'epochs': '4'}
```


**Training Jobs:**
I used 4 max jobs with 2 concurrent jobs.
It took 14 minutes to complete all 4 jobs, I will use 4 concurrent jobs next time to save time. 
![Training Jobs](https://github.com/vanusquarm/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/screenshots/training-jobs.PNG)

**Hyperparameter Training Jobs:**
![Hyperparameters Training Jobs](https://github.com/vanusquarm/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/screenshots/hyperparameter-training-jobs.PNG)

**Best Hyperparameters:**
![Hyperparameters](https://github.com/vanusquarm/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/screenshots/best-training-job.PNG)


## Debugging and Profiling
I first configured a debugger rule object that accepts a list of rules against output tensors that I want to evaluate. SageMaker Debugger automatically runs the ProfilerReport rule by default. This rules autogenerates a profiling report
Secondly, I configured a debugger hook parameter to adjust save intervals of the output tensors in the different training phases.
Next, I constructed a PyTorch estimator object with the debugger rule object and hook parameters.
I finally started the training job by fitting the training data to the estimator object.

### Results
My training job was quite short. Observing the peaks in utilization of cpu, gpu, memory and IO helped to better select the right instance type for training for improved resource efficiency.
However, I experienced a higher bottleneck in cpu operation indicting that the gpu was waiting most of the time for data to arrive 

![Hyperparameters](https://github.com/vanusquarm/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/screenshots/cloudwatch-logs.PNG)

## Model Deployment
The deployed model runs on 1 instance type of a standard compute resource ("ml.t2.medium"). The configuration of these parameters are set using the PyTorch deploy function. 
Upon performing the model deploy, an Endpoint is created. 
To query the endpoint with the test sample input, first perform a resize, crop, toTensor, and normalization transformation on the image, and then pass the transformed image to the predict function of the endpoint.

Execute the following lines of code replacing `IMAGE_PATH` by the path where your image is stored and `ENDPOINT` by the name of your endpoint:
```python
import io
import sagemaker
from PIL import Image
from sagemaker.serializers import IdentitySerializer
from sagemaker.pytorch.model import PyTorchPredictor

serializer = IdentitySerializer("image/jpeg")
predictor = PyTorchPredictor(ENDPOINT, serializer=serializer, sagemaker_session=sagemaker.Session())

buffer = io.BytesIO()
Image.open(IMAGE_PATH).save(buffer, format="JPEG")
response = predictor.predict(buffer.getvalue())
```

**ACTIVE ENDPOINT**
- SAGEMAKER STUDIO UI
![Active Endpoint](https://github.com/vanusquarm/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/blob/main/screenshots/active-endpoint.PNG)



