# Sentiment Analysis with Amazon SageMaker

The notebook and Python files provided here result in a simple web app which interacts with a deployed recurrent neural network performing sentiment analysis on movie reviews. This project assumes familiarity with Amazon SageMaker, XGBoost, PyTorch, and end-to-end machine learning pipelines.

The SageMaker_Project.ipynb Jupyter notebook contains detailed instructions for the end-to-end ML workflow from preparing the raw dataset to API deployment within Amazon SageMaker.

The 'train' folder contains model and training scripts.
The 'serve' folder contains model, inference, and util scripts.
The 'website' folder contains the front end code for the movie review web app.

Unit tests are provided in the tests folder.

Unit test coverage:

* Neural network layer input and output shapes
* Custom forward methods
* Custom training and validation loops
* Data preparation functions
* More

Follow these steps to build a Docker image for the training code:

* Open a new terminal using a SageMaker notebook instance. Make sure you are in the directory of the Dockerfile.

* Log into the ECR region of the base image (see Dockerfile) using the AWS CLI. This gives us permission to download the base PyTorch container, and it depends on your region: aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-2.amazonaws.com

* Create a repository in the AWS Elastic Container Registry (ECR) to store your image.

* Click on this repository and click 'View Push Commands' on the top right. Follow the 4 steps to upload your Docker image through the terminal. 

* To use this Docker image in SageMaker notebooks, create an Estimator object and pass in the image URI for your container in the image_name constructor parameter. This is how you would use built-in machine learning algorithms within SageMaker.
