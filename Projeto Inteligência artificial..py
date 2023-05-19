#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Extração de dados
import pandas as pd

tabela = pd.read_csv("barcos_ref.csv")
display(tabela)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(tabela.corr()[["Preco"]], annot=True, cmap="Blues")
plt.show()


# In[ ]:


# Modelagem e algoritmo

from sklearn.model_selection import train_test_split

# Separação da base em dados em X e Y
y = tabela["Preco"]

# Axis = 0 / --> Eixo das linhas 
# Axis = 1 / --> Eixo das colunas 

# Separando os dados de treino ou teste
x = tabela.drop("Preco", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


# In[ ]:


# Importa a inteligência artificial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Cria as inteligencias aritificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# Treina as inteligencias artificias
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# In[ ]:


from sklearn import metrics

# criar as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# comparar os modelos
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))  


# In[ ]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

# plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# In[ ]:


nova_tabela = pd.read_csv("novos_barcos.csv")
display(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)


# In[ ]:


get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')

