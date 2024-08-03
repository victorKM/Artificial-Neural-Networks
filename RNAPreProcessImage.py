import cv2

# Defindo um array com os nomes das pastas onde serão usadas as imagens
pastas = ['Areia', 'Grama', 'Madeira', 'Pedra']

for i in range(0,4):
  for j in range(0,25):
    
    # Carregar a imagem
    imagem = cv2.imread('ImagensPreProcessadas/' + pastas[i] + '/' + pastas[i] + str(j+1) + '.png')

    if imagem is not None:
      
      # Redimensionar a imagem para 256x256
      novoTamanho = (256, 256)
      imagemRedimensionada = cv2.resize(imagem, novoTamanho)

      # Salvar da imagem processada
      cv2.imwrite('ImagensProcessadas/' + pastas[i] + '/' + pastas[i] + str(j+1) + '.png', imagemRedimensionada)

    else:
      print('Pasta: ' + pastas[i] + ' | ' + 'Arquivo: ' + pastas[i] + str(j+1) + '.png')
      print("A imagem nao foi carregada.")