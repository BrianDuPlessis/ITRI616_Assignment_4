import tensorflow as tf
from tensorflow import keras

#Function to build the base neural network (which will be compared with the optimised neural network)
def create_model(labels, vectors, validation_vectors, validation_labels, epoc):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(28, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(vectors, labels, validation_data=(validation_vectors, validation_labels),epochs=epoc)
    return model

#Three optimisation profiles are provided

#tuner_model_1 optimises the following:
#-Number of hidden layers
#-Neurons per hidden layer
#-Activation function to use for each hidden layer
#-Learning rate
def tuner_model_1(hp):
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Int('Hidden_layers', 1, 6)):
        model.add(tf.keras.layers.Dense(units=hp.Int('Neurons_per_layer', min_value=32, max_value=512, step=128), activation=hp.Choice('activation_function' + str(i), ['relu', 'sigmoid'])))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#tuner_model_2 optimises the following:
#-Number of hidden layers
#-Neurons per hidden layer
#-Activation function to use for each hidden layer
def tuner_model_2(hp):
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Int('layers', 2, 6)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=50, max_value=100, step=10), activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile('adam','sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#tuner_model_3 optimises the following:
#-Neurons per hidden layer
#-Activation function to use for each hidden layer
#-Learning rate
def tuner_model_3(hp):
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=hp.Int('Neurons_layer_1', min_value=28, max_value=512, step=50), activation=hp.Choice('activation_function_1', ['relu', 'sigmoid'])))
    model.add(tf.keras.layers.Dense(units=hp.Int('Neurons_layer_2', min_value=28, max_value=512, step=50), activation=hp.Choice('activation_function_2', ['relu', 'sigmoid'])))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model