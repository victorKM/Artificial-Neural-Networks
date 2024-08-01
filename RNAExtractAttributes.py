import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

# Defindo um array com os nomes das pastas onde serão usadas as imagens
pastas = ['Areia', 'Grama', 'Madeira', 'Pedra']

# Defina o nome do arquivo de saída
arquivoOutput = "Attributes.txt"
atributoMatriz = []

for i in range(0,4):
  for j in range(0,25):

    # Carregar a imagem
    imagem = cv2.imread('ImagensProcessadas/' + pastas[i] + '/' + pastas[i] + str(j+1) + '.jpg', cv2.IMREAD_GRAYSCALE)

    if imagem is not None:
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

      # gerar um vetor [contraste, homogeinidade, energia, correlacao, media, desvioPadrao, entropia, classe]
      atributoVetor = [contraste, homogeinidade, energia,  correlacao, media,  desvioPadrao, entropia, pastas[i]]
      atributoMatriz.append(atributoVetor)
    else:
      print('Pasta: ' + pastas[i] + ' | ' + 'Arquivo: ' + pastas[i] + str(j+1) + '.jpg')
      print("A imagem nao foi carregada.")

with open(arquivoOutput, "w") as file:
    for i, vector in enumerate(atributoMatriz):
        file.write(", ".join(map(str, vector)))
        if i < len(atributoMatriz) - 1: 
            file.write("\n")