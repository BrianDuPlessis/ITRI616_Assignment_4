import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, classification_report, confusion_matrix

#Normalises pixel values and reshapes data into shape required by neural network
def preprocess_data(vector):
    vector = vector.reshape(vector.shape[0], -1) / 255.0
    vector = vector.reshape(-1, 28, 28)
    return vector

def basic_performance_evaluation(predicted_labels, actual_labels, testing_vectors):
    print("\nResults:")

    #Calculate performance metrics
    accuracy  = "{:.4f}".format(((predicted_labels == actual_labels).sum()/testing_vectors.shape[0])*100)
    precision = "{:.4f}".format(precision_score(actual_labels, predicted_labels, average='macro')*100)
    fOneScore = "{:.4f}".format(f1_score(actual_labels, predicted_labels, average='macro')*100)

    #Display performance metrics
    print("Model accuracy:\t\t" +accuracy+"%")
    print("Model precision:\t"  +precision+"%")
    print("Model f1-score:\t\t" +fOneScore+"%\n\n")

#Displays two confusion matrices side by side
def compare_models(actual_labels, base_predictions, optimised_predictions):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(actual_labels, base_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Base Model Confusion Matrix')

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(actual_labels, optimised_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Optimised Model Confusion Matrix')

    plt.tight_layout()
    plt.show()

def InDepth_performance_evaluation(actual_labels, predicted_labels):
    report = classification_report(actual_labels, predicted_labels)
    print("\nClassification Report\n")
    print(report)

def plot_confusion_matrix(cm, label):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {label}')
    plt.show()

def display_data_distibutions(training_labels, validation_labels, testing_labels):
    #Convert data format to allow data to be displayed and compared visually
    class_counts_training           = pd.Series(training_labels).value_counts().reset_index()
    class_counts_training.columns   = ['Category', 'Count']
    class_counts_validation         = pd.Series(validation_labels).value_counts().reset_index()
    class_counts_validation.columns = ['Category', 'Count']
    class_counts_testing            = pd.Series(testing_labels).value_counts().reset_index()
    class_counts_testing.columns    = ['Category', 'Count']

    #Generate visualisations
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.barplot(data=class_counts_training, x='Category', y='Count')
    plt.title('Training Data Distribution')

    plt.subplot(1, 3, 2)
    sns.barplot(data=class_counts_validation, x='Category', y='Count')
    plt.title('Validation Data Distribution')

    plt.subplot(1, 3, 3)
    sns.barplot(data=class_counts_testing, x='Category', y='Count')
    plt.title('Testing Data Distribution')

    plt.tight_layout()
    plt.show()