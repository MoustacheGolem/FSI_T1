# Universidade de Brasilia
# Departamento de Ciência da Computação
# Projeto 1 , Fundamentos de Sistemas Inteligentes,  2022/1

# Mateus de Paula Rodrigues - 190017953


import numpy as np
import pandas as pd

#sklearn usado pra montar a floresta em si
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

#************#
#*** MAIN ***#
#************#


RSEED = 6996
#'hepatitis.csv' contem ambos os dados que originalmente estavam separados para training e test
# eu vou separalos novalmente da maneira necessaria
df = pd.read_csv('hepatitis.csv').sample(155, random_state = RSEED)



## 1. Faça uma análise estatística inicial dos dados, plotando as quantidades médias, desvios padrões
## de todas as variáveis dos dados; (1,0 ponto)

out = df.replace({'?': np.nan})
print(out)
# 2. Construa um modelo de árvore de decisão (ID3, C4.5 ou CART), separando aleatoriamente
# sempre 10% dos dados para teste, em validação cruzada (com 10 rodadas), e mostre o resultado
# final em termos de: curva ROC, curva AUC ROC, e matriz de confusão. (2,0 pontos)


# 3. Construa um modelo de “floresta randômica”, com 100 árvores, usando todas as variáveis
# preditoras (i.e. m=9), separando aleatoriamente sempre 10% dos dados para teste, em validação
# cruzada (com 10 rodadas), e mostre o resultado final em termos de: curva ROC, curva AUC ROC, e
# matriz de confusão. (2,0 pontos)


# 4. Construa um modelo de “floresta randômica”, com 100 árvores, usando a raiz quadrada das
# variaǘeis preditoras (i.e. m=3), separando aleatoriamente sempre 10% dos dados para teste, em
# validação cruzada (com 10 rodadas), e mostre o resultado final em termos de: curva ROC, curva
# AUC ROC, e matriz de confusão. (2,0 pontos)


# 5. Mostre, para o caso do melhor resultado, quais as 2 mais importantes/relevantes variáveis
# preditoras. (1,0 ponto)
# 6. Gere, ou nos comentários do código, ou em um texto à parte as saídas e explicações pedidas no
# projeto. (2,0 pontos)