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

   ![image](https://github.com/user-attachments/assets/ea8358a5-e16f-48aa-a9f5-af2e3516693e)

5. **Testing with user**:
   - User can select images from computer and test if the class will be correct.

   ![image](https://github.com/user-attachments/assets/8183bbac-531e-4397-ae2a-6e947c77be9b)

   ![image](https://github.com/user-attachments/assets/523298e3-8b22-4cda-bf7c-38b8b1de1c4c)

## Technologies Used

- Python
- Libraries: scikit-learn, numpy, matplotlib, skimage, cv2, pickle, tkinter, PIL


