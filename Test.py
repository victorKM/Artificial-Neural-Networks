import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics  import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from joblib import load
import cv2
from skimage.feature import graycomatrix, graycoprops
import pickle
import tkinter as tk
from tkinter import messagebox, font
from tkinter import filedialog as fd
from PIL import Image, ImageTk 

FILE = ""

def show_custom_message(class_name, image_path):
    # Cria uma janela principal
    root = tk.Tk()
    root.withdraw() 

    # Cria uma nova janela para a mensagem
    message_window = tk.Toplevel()
    message_window.title("")

    # Define o tamanho da janela
    message_window.geometry("800x600")  # Aumentar o tamanho da janela para acomodar a imagem

    # Define a fonte e o tamanho do texto
    custom_font = font.Font(family="Helvetica", size=20, weight="bold")

    # Adiciona um rótulo com o texto
    label = tk.Label(message_window, text=f"Classe: {class_name}!", font=custom_font)
    label.pack(pady=10)

    # Carregar e exibir a imagem
    img = Image.open(image_path)
    img = img.resize((400, 400)) 
    img_tk = ImageTk.PhotoImage(img)

    img_label = tk.Label(message_window, image=img_tk)
    img_label.image = img_tk 
    img_label.pack(pady=10)

    # Botao de OK
    ok_button = tk.Button(
                    message_window,
                    text="OK",
                    command=message_window.destroy,
                    width=20,
                    height=2,
                    font=("Helvetica", 12, "bold")  # Fonte Helvetica, tamanho 12, negrito
                )
    ok_button.pack(pady=10)

    # Mantem janela aberta
    message_window.mainloop()
  

def main():
    np.set_printoptions(precision=13, suppress=True)

    # Carregar a MLP
    mlp = pickle.load(open("model", 'rb'))

    # Criar uma interface gráfica para selecionar a imagem
    root = tk.Tk()
    root.withdraw()
    file = fd.askopenfile()
    
    imagem = cv2.imread(file.name)

    if imagem is not None:

      # Redimensionar a imagem para 256x256
      novoTamanho = (256, 256)
      imagemRedimensionada = cv2.resize(imagem, novoTamanho)

      cv2.imwrite('imagem.png', imagemRedimensionada)
      imagem = cv2.imread('imagem.png', cv2.IMREAD_GRAYSCALE)
      
      # Iniciar extracao de atributos
      glcm = graycomatrix(imagem, [1], [0], symmetric=True, normed=True)

      # Extraia atributos da GLCM
      contraste = graycoprops(glcm, 'contrast')[0][0]
      homogeinidade = graycoprops(glcm, 'homogeneity')[0][0]
      energia = graycoprops(glcm, 'energy')[0][0]
      correlacao = graycoprops(glcm, 'correlation')[0][0]
      media = np.mean(imagem)
      desvioPadrao = np.std(imagem)
      hist, _ = np.histogram(imagem.flatten(), bins=256, range=[0,256])
      histSoma = np.sum(hist)
      epsilon = 1e-10
      entropia = -np.sum(np.nan_to_num((hist / (histSoma + epsilon)) * np.log2((hist + epsilon) / (histSoma + epsilon))))

      # Vetor de atributos
      atributos = np.array([contraste, homogeinidade, energia, correlacao, media, desvioPadrao, entropia])

      # Normalização Min-Max
      min_val = np.array([2.2135569852941,  0.0245327833519,  0.0064963714559,  0.0815281160685, 61.1517791748047,  6.4181027238808,  3.7718027076496])
      max_val = np.array([4079.592095588236,     0.7173791186012,    0.5334800277578, 0.9937278813403,  223.8517456054688,   71.6525687270341, 7.9456970125464])
      atributos_normalizados = 2 * ((atributos - min_val) / (max_val - min_val)) - 1

      classe = mlp.predict([atributos_normalizados])

      show_custom_message(classe[0], file.name)

    else:
      print("A imagem nao foi carregada.")
    
# Chamando main
if __name__ == "__main__":
    main()