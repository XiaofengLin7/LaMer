#!/bin/bash

if [ -z "$FLYTE_INTERNAL_EXECUTION_PROJECT" ]; then
    echo "This script only works when running in the grid cluster using Flyte. Do nothing."
    exit 0
fi

export MLFLOW_TRACKING_URI='http://mlflow-service.mlflow:5000'

# Get the last part of the workflow name as workspace name to avoid long project name
WORKSPACE_NAME=`echo $FLYTE_INTERNAL_EXECUTION_WORKFLOW | awk -F'.' '{print $(NF)}'`
export MLFLOW_EXPERIMENT_NAME="$FLYTE_INTERNAL_EXECUTION_PROJECT/$WORKSPACE_NAME"

echo "##### Automatic setting up MLFlow for HuggingFace Trainer #####"
echo "Please navigate to go/mlflowui and search for \"$MLFLOW_EXPERIMENT_NAME\" in \"experiments\" after the training started."
echo
