# Project Description

This project was developed as part of the Artificial Neural Networks course at the university. The main goal is to create and train a neural network to classify different types of image textures. The dataset includes four distinct texture types, each representing a different class. These textures were selected to have unique visual characteristics, which help in training and evaluating the model.

## Project Structure

The project is organized into the following main stages:

1. **Data Preprocessing**:
    - Reading and loading images from the files.
    - Normalizing the data to ensure values are within an appropriate range for neural network training.

2. **Model Definition**:
    - Using a Multilayer Perceptron (MLP) with adjustable hyperparameters such as activation function, number of neurons in each hidden layer, and learning rate.
    - Implementing cross-validation to robustly assess model performance.

3. **Training and Evaluation**:
    - Training the model on training data and evaluating it on test data for each fold during cross-validation.
    - Calculating performance metrics such as accuracy and confusion matrix for each fold and final average accuracy.

4. **Results Visualization**:
    - Generating confusion matrix plots to visualize classification performance for each texture type.

## Technologies Used

- Python
- Libraries: scikit-learn, numpy, matplotlib

## How to Run

To run the project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure all dependencies are installed (`scikit-learn`, `numpy`, `matplotlib`).
3. Run the main script to start the training and evaluation process.
