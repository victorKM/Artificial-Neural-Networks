import cv2

# Defindo um array com os nomes das pastas onde serão usadas as imagens
pastas = ['Areia', 'Grama', 'Madeira', 'Pedra']

for i in range(0,4):
  for j in range(0,25):
    
    # Carregar a imagem
    imagem = cv2.imread('ImagensPreProcessadas/' + pastas[i] + '/' + pastas[i] + str(j+1) + '.jpg')

    if imagem is not None:
      # Redimensionar a imagem para 256x256
      novoTamanho = (256, 256)
      imagemRedimensionada = cv2.resize(imagem, novoTamanho)

      # Converter imagem para tons de cinza
      imagemCinza = cv2.cvtColor(imagemRedimensionada, cv2.COLOR_BGR2GRAY)

      # Normalizar a imagem para ficar com valores de pixels entre 0 e 1
      imagemNormalizada = imagemCinza / 255.0

      # Equalizar a imagem
      imagemEqualizada = cv2.equalizeHist(imagemCinza)

      # Salvar da imagem processada
      cv2.imwrite('ImagensProcessadas/' + pastas[i] + '/' + pastas[i] + str(j+1) + '.jpg', imagemEqualizada)

    else:
      print('Pasta: ' + pastas[i] + ' | ' + 'Arquivo: ' + pastas[i] + str(j+1) + '.jpg')
      print("A imagem nao foi carregada.")