import os
import numpy as np
import pandas as pd
import NN_model as neuralNetwork
import evaluation_and_preprocessing 
from kerastuner.tuners import RandomSearch

#Find and load training and testing datasets
train_file_name             = "mnist_train.csv"
test_file_name              = "mnist_test.csv"
training_path               = os.path.dirname(__file__) + "\\" + train_file_name
testing_path                = os.path.dirname(__file__) + "\\" + test_file_name
training_validation_dataset = pd.read_csv(training_path)
testing_dataset             = pd.read_csv(testing_path)

#Training, testing, and validation size
total_records   = 70000;
training_size   = round(total_records*0.8)
validation_size = round(total_records*0.1)
testing_size    = round(total_records*0.1)

#Separate training, validation, and testing data and labels into separate variables 
training_vectors   = training_validation_dataset.iloc[:training_size, 1:].values
training_labels    = training_validation_dataset.iloc[:training_size, 0].values
validation_vectors = training_validation_dataset.iloc[training_size:training_size+validation_size, 1:].values
validation_labels  = training_validation_dataset.iloc[training_size:training_size+validation_size, 0].values
testing_vectors    = testing_dataset.iloc[:testing_size, 1:].values
testing_labels     = testing_dataset.iloc[:testing_size, 0].values

#Display shape of the data
print("\nShape of data")
print("training and validation dataset: \t"  + str(training_validation_dataset.shape))
print("testing dataset: \t\t\t"  + str(testing_dataset.shape))
print("training data: \t\t"  + str(training_vectors.shape))
print("validation data: \t"+ str(validation_vectors.shape))
print("testing data: \t\t"   + str(testing_vectors.shape))

#Generate visualisations to analyse data distribution of number categories
evaluation_and_preprocessing.display_data_distibutions(training_labels, validation_labels, testing_labels)

#Normalise pixel values, and reshape data
training_vectors   = evaluation_and_preprocessing.preprocess_data(training_vectors)
validation_vectors = evaluation_and_preprocessing.preprocess_data(validation_vectors)
testing_vectors    = evaluation_and_preprocessing.preprocess_data(testing_vectors)

#Build base neural network
print("\n================================Base model execution================================ \n")
base_model                  = neuralNetwork.create_model(training_labels, training_vectors, validation_vectors, validation_labels, 10)
model_output           = np.array(base_model.predict(testing_vectors))
base_model_predictions = np.argmax(model_output, axis=1)
evaluation_and_preprocessing.basic_performance_evaluation(base_model_predictions, testing_labels, testing_vectors)

#Declare tuner to search for optimal hyperparameters within the neural network
tuner = RandomSearch(
    neuralNetwork.tuner_model_3,
    objective='val_accuracy',
    max_trials=10,
    directory='Parameter_optimisation',
    project_name='Models'
)

#Execute tuner
print("\n================================Hyperparameter optimisation process================================ \n")
tuner.search(training_vectors, training_labels,  epochs=10, validation_data=(validation_vectors, validation_labels))
tuner.results_summary()

#Rebuild neural network model with the optimised hyperparameters
print("\n======================================Optimised model execution==================================== \n")
best_parameters = tuner.get_best_hyperparameters(num_trials=1)[0]
opt_model           = tuner.hypermodel.build(best_parameters)
opt_model.fit(training_vectors, training_labels, validation_data=(validation_vectors, validation_labels), epochs=10)

#Make predictions and place prediction labels in an array for evaluation
model_output                = np.array(opt_model.predict(testing_vectors))
optimised_model_predictions = np.argmax(model_output, axis=1)
evaluation_and_preprocessing.basic_performance_evaluation(optimised_model_predictions, testing_labels, testing_vectors)

#Display base model parameters and performance figures
print("\n\n======================================Base model evaluation==================================== \n")
base_model.summary()
print("\n")
evaluation_and_preprocessing.InDepth_performance_evaluation(testing_labels, base_model_predictions)

#Display optimised model parameters and performance figures
print("\n\n====================================Optimised model evaluation================================== \n")
opt_model.summary()
print("\n")
evaluation_and_preprocessing.InDepth_performance_evaluation(testing_labels, optimised_model_predictions)

#Compare base and optimised neural network models
evaluation_and_preprocessing.compare_models(testing_labels, base_model_predictions, optimised_model_predictions)