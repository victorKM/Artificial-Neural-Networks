import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics  import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def main():
  # Lê os dados do arquivo
  linhas = carregarDados('Attributes.txt')

  # Manda os dados do arquivo para um array 
  dados = [list(map(float, linha.strip().split(',')[:-1])) + [linha.strip().split(',')[-1].strip()] for linha in linhas]

  # Normalizacao
  dadosNormalizados = normalizacao(dados,-1,1)
  
  # Divide os dados em treino e teste
  atributosNormalizados, classes = divisaoAtributoClasse(dadosNormalizados)

  # Coletar dados do usuário sobre a configuracao do MLP
  funcaoAtivacao, neuroniosCamadaOculta, taxaAprendizado = obterHiperparametros()

  # Criar a MLP
  mlp = MLPClassifier(hidden_layer_sizes=neuroniosCamadaOculta, activation=funcaoAtivacao, solver="adam",
                      learning_rate_init=taxaAprendizado, max_iter=2000, random_state=42)

  # Validação cruzada
  folds = int(input("Informe o número de folds que deseja: "))
  while(folds < 1):
     folds = int(input("Informe o novamente número de folds que deseja: "))
  cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

  # Inicializar lista para armazenar as acurácias de cada fold
  acuracias = []
  
  # Treinar e testar para cada fold
  for i, (indexTreino, indexTeste) in enumerate(cv.split(atributosNormalizados, classes)):
    print(f"\nFold {i + 1}:")

    # Dividir dados em treino e teste para este fold
    atributosTreinoFold, atributosTesteFold = atributosNormalizados[indexTreino], atributosNormalizados[indexTeste]
    classesTreinoFold, classesTesteFold = classes[indexTreino], classes[indexTeste]

    # Treinar a MLP para este fold
    mlp.fit(atributosTreinoFold, classesTreinoFold)

    # Testar
    classesPrevistasFold = mlp.predict(atributosTesteFold)

    # Calcular métricas para este fold
    acuracia = accuracy_score(classesTesteFold, classesPrevistasFold)
    print("Acuracia:", acuracia)
    acuracias.append(acuracia)

    matrizConfusao = confusion_matrix(classesTesteFold, classesPrevistasFold, labels=np.unique(classes))
    display = ConfusionMatrixDisplay(confusion_matrix=matrizConfusao, display_labels=np.unique(classes))
    display.plot()
    plt.show()

  # Calcular e imprimir a acurácia média final
  acuracia_media_final = np.mean(acuracias)
  print("\nAcurácia Média Final:", acuracia_media_final)

def carregarDados(nomeTxt):
  # Lendo os dados do arquivo
  with open(nomeTxt, 'r') as file:
    linhas = file.readlines()
  return linhas

def normalizacao(dados, rangeMin, rangeMax):
  # Separa os atributos e as classes
  atributos = np.array(dados)[:, :-1]
  classes = np.array(dados)[:, -1]

  # Normaliza os atributos entre -1 e 1
  scaler = MinMaxScaler(feature_range=(rangeMin, rangeMax))
  atributosNormalizados = scaler.fit_transform(atributos)

  # Combina os atributos normalizados com as classes (mantendo a última coluna como string)
  dadosNormalizados = np.column_stack((atributosNormalizados, classes))
  return dadosNormalizados

def divisaoAtributoClasse(dadosNormalizados):
  # Extrai atributos normalizados e classes
  atributosNormalizados = np.array(dadosNormalizados)[:, :-1].astype(float)
  classes = np.array(dadosNormalizados)[:, -1]

  return atributosNormalizados, classes

def obterHiperparametros():
   funcaoAtivacao = obterFuncaoAtivacao()
   neuroniosCamadaOculta = obterNeuroniosCamadaOculta()
   taxaAprendizado = obterTaxaAprendizado()
   return funcaoAtivacao, neuroniosCamadaOculta, taxaAprendizado
  
def obterFuncaoAtivacao():
  opcoesFuncaoAtivacao = ["identity", "logistic", "tanh", "relu"]
  opcaoEscolhida = input("Escolha uma funcao de ativacao (1) Identidade (2) Logística (3) Tangente (4) Relu: ")

  funcaoAtivacao = None
  while funcaoAtivacao is None:
    opcaoEscolhida = int(opcaoEscolhida)
    if(opcaoEscolhida >= 1 and opcaoEscolhida <= 4):
      funcaoAtivacao = opcoesFuncaoAtivacao[opcaoEscolhida-1]
    else: 
      opcaoEscolhida = input("Escolha uma opcao valida: ")
  return funcaoAtivacao

def obterNeuroniosCamadaOculta():
  numeroCamadaOculta = int(input("Escolha o numero de camadas ocultas no MLP: "))
  neuroniosCamadaOculta = []
  for i in range(0, numeroCamadaOculta):
    numeroNeuronioCamada = int(input("Escolha o numero de neuronios para a " + str(i+1) + "a camada oculta: "))
    neuroniosCamadaOculta.append(numeroNeuronioCamada)
  return tuple(neuroniosCamadaOculta)

def obterTaxaAprendizado():
  taxaAprendizado = float(input("Forneca o valor da taxa de aprendizado entre 0.001 e 1: "))
  while(taxaAprendizado < 0.001 or taxaAprendizado > 1):
    taxaAprendizado = float(input("Forneca o valor correto da taxa de aprendizado entre 0.1 e 1: "))
  return taxaAprendizado

# Chamando main
if __name__ == "__main__":
    main()