import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import pickle
import tkinter as tk
from tkinter import font
from tkinter import filedialog as fd
from PIL import Image, ImageTk 

def __main__():
    # Set number places after decimal point
    np.set_printoptions(precision=13, suppress=True)

    # Load MLP
    mlp = pickle.load(open("MLP/model", 'rb'))

    # Create interface to select the image
    root = tk.Tk()
    root.withdraw()
    file = fd.askopenfile()
    
    image = cv2.imread(file.name)

    if image is not None:
      image = cv2.resize(image, (256, 256))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Begin attributes extraction
      glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)

      # Extract attributes from GLCM
      contrast = graycoprops(glcm, 'contrast')[0][0]
      homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
      energy = graycoprops(glcm, 'energy')[0][0]
      correlation = graycoprops(glcm, 'correlation')[0][0]
      average = np.mean(image)
      standard_deviation = np.std(image)
      hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])
      histSum = np.sum(hist)
      epsilon = 1e-10
      entropy = -np.sum(np.nan_to_num((hist / (histSum + epsilon)) * np.log2((hist + epsilon) / (histSum + epsilon))))

      # Attributes vector
      attributes = np.array([contrast, homogeneity, energy, correlation, average, standard_deviation, entropy])

      # Normalizations
      with open('MLP/MaxMin.txt', 'r') as text:
          lines = text.readlines()
      data = [list(map(float, line.strip().split(','))) for line in lines]

      min_val = np.array(data[0]).astype(float)
      max_val = np.array(data[1]).astype(float)
      normalized_attributes = 2 * ((attributes - min_val) / (max_val - min_val)) - 1

      class_name = mlp.predict([normalized_attributes])

      show_custom_message(class_name[0], file.name)

    else:
      print("Image was not loaded.")

def show_custom_message(class_name, image_path):
    # Create new window for message
    message_window = tk.Toplevel()
    message_window.title("")

    # Window size
    message_window.geometry("800x600") 
    
    # Font and text size
    custom_font = font.Font(family="Helvetica", size=20, weight="bold")

    label = tk.Label(message_window, text=f"Class: {class_name}!", font=custom_font)
    label.pack(pady=10)

    # Show image
    img = Image.open(image_path)
    img = img.resize((400, 400))
    img_tk = ImageTk.PhotoImage(img)

    img_label = tk.Label(message_window, image=img_tk)
    img_label.image = img_tk
    img_label.pack(pady=10)

    # OK button
    ok_button = tk.Button(
                    message_window,
                    text="OK",
                    command=message_window.destroy,
                    width=20,
                    height=2,
                    font=("Helvetica", 12, "bold")
                )
    ok_button.pack(pady=10)

    # Wait until window close
    message_window.wait_window()
    
if __name__ == "__main__":
    __main__()