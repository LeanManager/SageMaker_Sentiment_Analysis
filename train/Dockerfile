FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.4.0-gpu-py36-cu101-ubuntu16.04

RUN pip install sagemaker-containers

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py