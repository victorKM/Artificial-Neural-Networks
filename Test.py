import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import pickle
import tkinter as tk
from tkinter import messagebox, font
from tkinter import filedialog as fd
from PIL import Image, ImageTk 
import os

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
    mlp = pickle.load(open("MLP/model", 'rb'))

    # Criar uma interface gráfica para selecionar a imagem
    root = tk.Tk()
    root.withdraw()
    file = fd.askopenfile()
    
    imagem = cv2.imread(file.name)

    if imagem is not None:

      # Redimensionar a imagem para 256x256
      novoTamanho = (256, 256)
      imagemRedimensionada = cv2.resize(imagem, novoTamanho)

  	  # Armazena temporariamente a imagem
      cv2.imwrite('imagem.png', imagemRedimensionada)
      imagem = cv2.imread('imagem.png')
      os.remove('imagem.png')

      # Converte para escalas de cinza
      imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

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

      # Normalização com MinMax
      with open('MLP/MaxMin.txt', 'r') as text:
          linhas = text.readlines()
      dados = [list(map(float, linha.strip().split(','))) for linha in linhas]

      min_val = np.array(dados[0]).astype(float)
      max_val = np.array(dados[1]).astype(float)
      atributos_normalizados = 2 * ((atributos - min_val) / (max_val - min_val)) - 1

      classe = mlp.predict([atributos_normalizados])

      show_custom_message(classe[0], file.name)

    else:
      print("A imagem nao foi carregada.")
    
# Chamando main
if __name__ == "__main__":
    main()