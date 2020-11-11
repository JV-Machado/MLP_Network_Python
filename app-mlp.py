import pandas as pd

from mlp import MLP

dataset = pd.read_csv('database/treinamento.csv')
X = dataset.iloc[:, 0:4].values
d = dataset.iloc[:, 4:7].values

mlp = MLP(X,d, [15, 3])

mlp.train()

datasetTeste = pd.read_csv('database/teste.csv')
xteste = datasetTeste.iloc[:, 0:4].values
dteste = datasetTeste.iloc[:, 4:7].values

 
print()
print(">>>>>>TESTE<<<<<<")
print(f'Saída esperada: {datasetTeste.iloc[0,4:7].values}, Saída obtida: {mlp.evaluate([0.8622,0.7101,0.6236,0.7894])}') 
print(f'Saída esperada: {datasetTeste.iloc[1,4:7].values}, Saída obtida: {mlp.evaluate([0.2741,0.1552,0.1333,0.1516])}') 
print(f'Saída esperada: {datasetTeste.iloc[2,4:7].values}, Saída obtida: {mlp.evaluate([0.6772,0.8516,0.6543,0.7573])}') 
print(f'Saída esperada: {datasetTeste.iloc[3,4:7].values}, Saída obtida: {mlp.evaluate([0.2178,0.5039,0.6415,0.5039])}') 
print(f'Saída esperada: {datasetTeste.iloc[4,4:7].values}, Saída obtida: {mlp.evaluate([0.726,0.75,0.7007,0.4953])}') 
print(f'Saída esperada: {datasetTeste.iloc[5,4:7].values}, Saída obtida: {mlp.evaluate([0.2473,0.2941,0.4248,0.3087])}') 
print(f'Saída esperada: {datasetTeste.iloc[6,4:7].values}, Saída obtida: {mlp.evaluate([0.5682,0.5683,0.5054,0.4426])}') 
print(f'Saída esperada: {datasetTeste.iloc[7,4:7].values}, Saída obtida: {mlp.evaluate([0.6566,0.6715,0.4952,0.3951])}')
print(f'Saída esperada: {datasetTeste.iloc[8,4:7].values}, Saída obtida: {mlp.evaluate([0.0705,0.4717,0.2921,0.2954])}') 
print(f'Saída esperada: {datasetTeste.iloc[9,4:7].values}, Saída obtida: {mlp.evaluate([0.1187,0.2568,0.314,0.3037])}') 
print(f'Saída esperada: {datasetTeste.iloc[10,4:7].values}, Saída obtida: {mlp.evaluate([0.5673,0.7011,0.4083,0.5552])}') 
print(f'Saída esperada: {datasetTeste.iloc[11,4:7].values}, Saída obtida: {mlp.evaluate([0.3164,0.2251,0.3526,0.256])}')
print(f'Saída esperada: {datasetTeste.iloc[12,4:7].values}, Saída obtida: {mlp.evaluate([0.7884,0.9568,0.6825,0.6398])}')
print(f'Saída esperada: {datasetTeste.iloc[13,4:7].values}, Saída obtida: {mlp.evaluate([0.9633,0.785,0.6777,0.6059])}')
print(f'Saída esperada: {datasetTeste.iloc[14,4:7].values}, Saída obtida: {mlp.evaluate([0.7739,0.8505,0.7934,0.6626])}')
print(f'Saída esperada: {datasetTeste.iloc[15,4:7].values}, Saída obtida: {mlp.evaluate([0.4219,0.4136,0.1408,0.094])}')
print(f'Saída esperada: {datasetTeste.iloc[16,4:7].values}, Saída obtida: {mlp.evaluate([0.6616,0.4365,0.6597,0.8129])}')
print(f'Saída esperada: {datasetTeste.iloc[17,4:7].values}, Saída obtida: {mlp.evaluate([0.7325,0.4761,0.3888,0.5683])}')
print()
