import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def __main__():
  # Array with pasts name
  pasts = ['Sand', 'Grass', 'Wood', 'Stone']

  # Output file name
  fileOutput = "MLP/Attributes.txt"
  attributeMatrix = []

  for i in range(0,4):
    for j in range(0,25):

      #Load the image
      image = cv2.imread('PreProcessedImages/' + pasts[i] + '/' + pasts[i] + str(j+1) + '.png')
      image = cv2.resize(image, (256,256))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      if image is not None:
        glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)

        # Extract attributes from GLCM
        constrast = graycoprops(glcm, 'contrast')[0][0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
        energy = graycoprops(glcm, 'energy')[0][0]
        correlation = graycoprops(glcm, 'correlation')[0][0]
        average = np.mean(image)
        standard_deviation = np.std(image)
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0,256])
        histSum = np.sum(hist)
        epsilon = 1e-10
        entropy = -np.sum(np.nan_to_num((hist / (histSum + epsilon)) * np.log2((hist + epsilon) / (histSum + epsilon))))

        atributoVetor = [constrast, homogeneity, energy,  correlation, average,  standard_deviation, entropy, pasts[i]]
        attributeMatrix.append(atributoVetor)
      else:
        print('Past: ' + pasts[i] + ' | ' + 'File: ' + pasts[i] + str(j+1) + '.png')
        print("Image was not loaded.")

  with open(fileOutput, "w") as file:
      for i, vector in enumerate(attributeMatrix):
          file.write(", ".join(map(str, vector)))
          if i < len(attributeMatrix) - 1: 
              file.write("\n")

if __name__ == "__main__":
    __main__()