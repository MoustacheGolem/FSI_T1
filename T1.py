# Universidade de Brasilia
# Departamento de Ciência da Computação
# Projeto 1 , Fundamentos de Sistemas Inteligentes,  2022/1

# Mateus de Paula Rodrigues - 190017953


import numpy as np
import pandas as pd

#sklearn usado pra montar a floresta em si

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


#usadas pra montar o plot da confusion matrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools   


#************#
#*** MAIN ***#
#************#

RSEED = 6996
#'hepatitis.csv' contem ambos os dados que originalmente estavam separados para training e test
# eu vou separalos novalmente da maneira necessaria
df = pd.read_csv('hepatitis.csv').sample(155, random_state = RSEED)



## 1. Faça uma análise estatística inicial dos dados, plotando as quantidades médias, desvios padrões
## de todas as variáveis dos dados; (1,0 ponto)

outcome = np.array(df.pop('CLASS')) # Separa as features da label com o resultado
df = df.replace({'?': np.nan}).astype(float) #substitui '?' por na

results = []
for col in df:
    results.append([col,"%.2f" % df[col].mean(),"%.2f" % df[col].std()])
    

print(f"{'Nome:' : <15} | {'Média' : ^10} | {'DesvioPadrão' : >10}") 
print("-------------------------------------------")
for i in results:
    print(f"{i[0] : <15} | {i[1] : ^10} | {i[2] : >10}") 


def evaluate_model(model,modo):
    
    # """
    # essa funcao foi parcialmente baseada na funcao evaluate_model
    # source: https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb
    # """
    
    #######################################################
    # calcula e printa os valores                         #
    #######################################################
    if(modo):
        n_nodes = []
        max_depths = []

        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        print('Numero medio de nodos: ', int(np.mean(n_nodes)))
        print('Media de profundidade: ', int(np.mean(max_depths)))



    
    #codigo responsavel por plotar um grafo ROC
    
    base_fpr, base_tpr, _ = roc_curve(test_outcome, [1 for _ in range(len(test_outcome))], pos_label=2)
    model_fpr, model_tpr, _ = roc_curve(test_outcome, probs,pos_label=2)

    plt.figure(figsize = (6, 4))
    plt.rcParams['font.size'] = 16
    
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # """
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    # Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    

    plt.figure(figsize = (5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('Outcome Verdadeira', size = 18)
    plt.xlabel('Outcome Prevista', size = 18)
    plt.show()


# 2. Construa um modelo de árvore de decisão (ID3, C4.5 ou CART), separando aleatoriamente
# sempre 10% dos dados para teste, em validação cruzada (com 10 rodadas), e mostre o resultado
# final em termos de: curva ROC, curva AUC ROC, e matriz de confusão. (2,0 pontos)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#separa os dados 
train, test, train_outcome, test_outcome  = train_test_split(df, outcome,
                                                            stratify = outcome,
                                                            test_size = 0.1,
                                                            random_state = RSEED)
#limpar nan
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Guardar Features pra parte 5
features = list(train.columns)

# monta arvore
arvore = DecisionTreeClassifier(random_state=RSEED)
arvore.fit(train, train_outcome)

# validacao cruzada
param_grid = {
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
}
rs = RandomizedSearchCV(arvore, param_grid, n_jobs = 1, 
                        scoring = 'roc_auc', cv = 10, 
                        n_iter = 10, verbose = 0, random_state=RSEED)
rs.fit(train, train_outcome)
arvore = rs.best_estimator_

# printa tamanho da arvore.
print(f'Arvore tem: {arvore.tree_.node_count} nodes com profundidade maxima de: {arvore.tree_.max_depth}.')


# coletar dados
train_probs = arvore.predict_proba(train)[:, 1]
probs = arvore.predict_proba(test)[:, 1]

train_predictions = arvore.predict(train)
predictions = arvore.predict(test)



# printa score ROC
print(f'Train ROC AUC Score: {roc_auc_score(train_outcome, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_outcome, probs)}')

# plot curva ROC
evaluate_model(arvore,False)


# plot matriz de confusão
cm = confusion_matrix(test_outcome, predictions)
plot_confusion_matrix(cm, classes = ['Class 1', 'Class 2'],
                       title = 'Hepatitis Confusion Matrix')

# 3. Construa um modelo de “floresta randômica”, com 100 árvores, usando todas as variáveis
# preditoras (i.e. m=9)19, separando aleatoriamente sempre 10% dos dados para teste, em validação
# cruzada (com 10 rodadas), e mostre o resultado final em termos de: curva ROC, curva AUC ROC, e
# matriz de confusão. (2,0 pontos)
n_arvores = 100
floresta = RandomForestClassifier(n_estimators = n_arvores, 
                               random_state=RSEED, 
                               max_features = 19,
                               n_jobs=-1, verbose = 0)

# Treinar floresta
floresta.fit(train, train_outcome)

param_grid = {
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}


# Modelo de busca aleatoria, para validação cruzada.
rs = RandomizedSearchCV(floresta, param_grid, n_jobs = 1, 
                        scoring = 'roc_auc', cv = 10, 
                        n_iter = 10, verbose = 0, random_state=RSEED)

# Trainar modelo 
rs.fit(train, train_outcome)
best_forest = rs.best_estimator_

train_probs = best_forest.predict_proba(train)[:, 1]
probs = best_forest.predict_proba(test)[:, 1]

train_predictions = best_forest.predict(train)
predictions = best_forest.predict(test)


print(f'Train ROC AUC Score: {roc_auc_score(train_outcome, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_outcome, probs)}')

# plot curva ROC
evaluate_model(best_forest,True)


# plot matriz de confusão
cm = confusion_matrix(test_outcome, predictions)
plot_confusion_matrix(cm, classes = ['Class 1', 'Class 2'],
                       title = 'Hepatitis Confusion Matrix')



# 4. Construa um modelo de “floresta randômica”, com 100 árvores, usando a raiz quadrada das
# variaǘeis preditoras (i.e. m=3), separando aleatoriamente sempre 10% dos dados para teste, em
# validação cruzada (com 10 rodadas), e mostre o resultado final em termos de: curva ROC, curva
# AUC ROC, e matriz de confusão. (2,0 pontos)

n_arvores = 100
floresta2 = RandomForestClassifier(n_estimators = n_arvores, 
                               random_state=RSEED, 
                               max_features = 4,  #Unica modificacao, 4 equivalente a 'sqrt' aqui.
                               n_jobs=-1, verbose = 0)

#treinar floresta 2
floresta2.fit(train, train_outcome)

param_grid = {
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

# Modelo de busca aleatoria, para validação cruzada.
rs2 = RandomizedSearchCV(floresta2, param_grid, n_jobs = 1, 
                        scoring = 'roc_auc', cv = 10, 
                        n_iter = 10, verbose = 0, random_state=RSEED)

# Trainar modelo  
rs2.fit(train, train_outcome)
best_forest2 = rs2.best_estimator_

train_probs = best_forest2.predict_proba(train)[:, 1]
probs = best_forest2.predict_proba(test)[:, 1]

train_predictions = best_forest2.predict(train)
predictions = best_forest2.predict(test)


print(f'Train ROC AUC Score: {roc_auc_score(train_outcome, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_outcome, probs)}')

# plot curva ROC
evaluate_model(best_forest2,True)



# plot matriz de confusão
cm2 = confusion_matrix(test_outcome, predictions)
plot_confusion_matrix(cm2, classes = ['Class 1', 'Class 2'],
                       title = 'Hepatitis Confusion Matrix')


# 5. Mostre, para o caso do melhor resultado, quais as 2 mais importantes/relevantes variáveis
# preditoras. (1,0 ponto)

features = list(train.columns)   
fi_model = pd.DataFrame({'feature': features,
                'importance': floresta2.feature_importances_}).\
                sort_values('importance', ascending = False)
                
print('\n',fi_model.head(8))

