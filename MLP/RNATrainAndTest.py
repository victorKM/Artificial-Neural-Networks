import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics  import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pickle

def __main__():
  # Set number places after decimal point
  np.set_printoptions(precision=13, suppress=True)

  # Read file data
  lines = load_data('MLP/Attributes.txt')

  # Send data to array
  data = [list(map(float, line.strip().split(',')[:-1])) + [line.strip().split(',')[-1].strip()] for line in lines]

  # Normalization
  normalized_data = normalization(data,-1,1)

  # Split in train and test data
  normalized_attributes, classes = split_data(normalized_data)

  # Collect user data for configure MLP
  activation_function, neurons_number, learning_rate = get_hyperparameters()

  # Create MLP
  mlp = MLPClassifier(hidden_layer_sizes=neurons_number, activation=activation_function, solver="adam",
                      learning_rate_init=learning_rate, max_iter=2000, random_state=42)

  # Cross Validation
  folds = int(input("Type the number of folds that you want: "))
  while(folds < 1):
     folds = int(input("Type again the number of folds that you want: "))
  cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

  # List to store folders accuracies
  accuracies = []
  
  # Train and test for each fold
  for i, (train_index, test_index) in enumerate(cv.split(normalized_attributes, classes)):
    print(f"\nFold {i + 1}:")

    # Split in train and test data for this fold
    train_fold_attributes, test_fold_attributes = normalized_attributes[train_index], normalized_attributes[test_index]
    train_fold_classes, test_fold_classes = classes[train_index], classes[test_index]

    # Train MLP for this fold
    mlp.fit(train_fold_attributes, train_fold_classes)

    # Test
    predict_fold_classes = mlp.predict(test_fold_attributes)

    # Calculate metrics for this fold
    accuracy = accuracy_score(test_fold_classes, predict_fold_classes)
    print("Accuracy:", accuracy)
    accuracies.append(accuracy)

    confusion_matr = confusion_matrix(test_fold_classes, predict_fold_classes, labels=np.unique(classes))
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matr, display_labels=np.unique(classes))
    display.plot()
    plt.show()

  # Create model file to test with user
  pickle.dump(mlp, open("MLP/model", 'wb'))

  # Calculate and show final average accuracy
  final_average_accuracy = np.mean(accuracies)
  print("\nFinal Average Accuracy:", final_average_accuracy)

def load_data(nomeTxt):
  # Read file data
  with open(nomeTxt, 'r') as file:
    lines = file.readlines()
  return lines

def normalization(data, rangeMin, rangeMax):
  # Separate attributes and classes
  attributes = np.array(data)[:, :-1].astype(float)
  classes = np.array(data)[:, -1]

  # Store max and min values in txt to use in test.py
  max = np.max(attributes, axis=0)
  min = np.min(attributes, axis=0)
  max_values = ', '.join(f"{val:.13f}" for val in max)
  min_values = ', '.join(f"{val:.13f}" for val in min)

  with open("MLP/MaxMin.txt", "w") as file:
    file.write(min_values)
    file.write("\n")
    file.write(max_values)

  # Normalize attributes between -1 and 1
  scaler = MinMaxScaler(feature_range=(rangeMin, rangeMax))
  normalized_attributes = scaler.fit_transform(attributes)

  # Bring together attributes and classes in on variable
  normalized_data = np.column_stack((normalized_attributes, classes))
  return normalized_data

def split_data(normalized_data):
  # Extract normalized attributes and classes
  normalized_attributes = np.array(normalized_data)[:, :-1].astype(float)
  classes = np.array(normalized_data)[:, -1]

  return normalized_attributes, classes

def get_hyperparameters():
   activation_function = get_activation_function()
   neurons_number = get_neurons_number()
   learning_rate = get_learning_rate()
   return activation_function, neurons_number, learning_rate
  
def get_activation_function():
  activate_function_options = ["identity", "logistic", "tanh", "relu"]
  option = input("Choose one activation function (1) Identity (2) Logistic (3) Tangent (4) Relu: ")

  activation_function = None
  while activation_function is None:
    option = int(option)
    if(option >= 1 and option <= 4):
      activation_function = activate_function_options[option-1]
    else: 
      option = input("Choose a valid option: ")
  return activation_function

def get_neurons_number():
  hidden_layers_number = int(input("Choose a number of hidden layers for MLP: "))
  neurons_number = []
  for i in range(0, hidden_layers_number):
    neurons_layer_number = int(input("Choose a neurons number for the " + str(i+1) + " hidden layer: "))
    neurons_number.append(neurons_layer_number)
  return tuple(neurons_number)

def get_learning_rate():
  learning_rate = float(input("Write learning rate between 0.001 and 1: "))
  while(learning_rate < 0.001 or learning_rate > 1):
    learning_rate = float(input("Write a correct learning rate between 0.001 and 1: "))
  return learning_rate

if __name__ == "__main__":
    __main__()