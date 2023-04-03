# Prediction of COVID-19 by performing binary classification on chest X-Ray images with Deep Leaning

WIth the use of Tensorflow and the Functional API, a convolutional neural network model is built in order to help in COVID-19 detection in a way that it classifies chest X-Ray images as positive or negative COVID-19 cases. The input dataset is composed of 33920 chest X-Ray images (<a href="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu"> Kaggle's COVID-QU-Ex Dataset </a>) and are divided into training (21715 images), validation (5417 images) and test (6788 images). The images are composed of COVID-19 cases, other virus and bacterial pneumonia and healthy chest images. The model managed to obtain an accuracy of 97%, precision of 95% and recall of 96% when submitted to the test set and so has presented to be very reliable in order to correctly predict positive COVID-19 cases and also in order to avoid false negative predictions.

## Contents:
* data: txt file informing where to download the data.
* models: Checkpoint folder with the trained model, history folder with the model history.
* notebooks: Notebooks for exploratory data analysis and to plot model metrics and results. Also contains a python script with plot and metric functions.
* output: figure and txt with model results and analysis.
* src: folder with configuration yaml file, python main script to create, train and save the model.