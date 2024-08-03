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

   - Generating confusion matrix plots to visualize classification performance for each texture type and for each fold.

   ![imagem1](https://github.com/user-attachments/assets/de530da8-b1d4-45f0-9ef8-00c58f2435db)


5. **Testing with user**:
   - User can select images from computer and test if the class will be correct.

   ![imagem2](https://github.com/user-attachments/assets/a3b0af07-c3c1-4069-9d87-103903ff7638)


   ![imagem3](https://github.com/user-attachments/assets/7011d18c-bb0f-4b63-a180-9f3a36451314)

## Technologies Used

- Python
- Libraries: scikit-learn, numpy, matplotlib


