# Distributed Training

__Distributed implementation for training a Image Classifier__

## Installation and Setup

### Prerequisites
- Python 2.7
- Pip
- Virtualenv
- GCloud SDK and Google Platform Account (Make sure to login to GCloud Account by the following command `gcloud auth application-default login`)

### Setup virtualenv
__Create virtual environment:__
- Mac/Linux: `virtualenv env`
- Windows: `virtualenv env`

__Activate virtual environment (Must activate before running any python scripts):__
- Mac/Linux: `source ./env/bin/activate`
- Windows: `.\env\Scripts\activate`

### Modules
__Install the following required modules:__
- `pip install --upgrade -r requirements.txt`

### Declare Variables
```
export LOCAL_PATH=<datapath-to-save-dataset>
export REGION=<your-region-for-google-cloud>
export BUCKET_NAME=<your-bucket-name>
gsutil mb gs://${BUCKET_NAME}/
export BUCKET=gs://${BUCKET_NAME}/
```

### Retrieve Dataset (Local and Cloud)
Use `data.py` script to save the dataset to a local datapath
```
python trainer/data.py \
--output_path ${LOCAL_PATH}
```
Save the dataset to the cloud by copying the dataset to the storage bucket
```
gsutil -m cp -r ${LOCAL_PATH}/tf_flowers ${BUCKET}/tf_flowers
```

## Train Model
__Activate virtual environment:__
Needed if you are training locally
- Mac/Linux: `source ./env/bin/activate`
- Windows: `.\env\Scripts\activate`

__Optional Arguments:__
These arguments can be used when training locally or on the cloud
- `num-epochs=<value>`
- `batch-size=<value>`
- `learning-rate=<value>`

### Train Locally
```
export DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_DIR=flower_${DATE}

python -m trainer.task \
--data-path=${LOCAL_PATH} \
--job-dir=${JOB_DIR} \
--num-epochs=<value> \
--batch-size=<value> \
--learning-rate=<value>
```

### Train in GCloud AI Platform 
#### Setup Cloud Config
Define the cloud configuration for the training job by modifying `config.yaml` to specify the number of instances, refer to [this](https://cloud.google.com/ml-engine/docs/tensorflow/machine-types) to define your custom configuration

#### Submit training job to GCloud AI Platform
```
export DATE=`date '+%Y%m%d_%H%M%S'` 
export JOB_NAME=flower_${DATE}  
export GCS_JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}  

gcloud ai-platform jobs submit training $JOB_NAME --stream-logs --runtime-version 1.10 \
--job-dir=${GCS_JOB_DIR} \
--package-path=trainer \
--module-name trainer.task \
--region=${REGION} \
--config config.yaml \
-- \
--data-path=${BUCKET} \
--num-epochs=<value> \
--batch-size=<value> \
--learning-rate=<value>
```

__Deactivate virtual environment when done by: `deactivate`__