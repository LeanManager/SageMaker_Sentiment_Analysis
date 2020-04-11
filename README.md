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
