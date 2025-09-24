import pandas as pd
import numpy as np 
import matplotlib . pyplot as plt 
import seaborn
from sklearn . decomposition import PCA 
from sklearn . preprocessing import StandardScaler


# Lire le fichier CSV
df = pd.read_csv('imdb_top_1000.csv')


# Supprimer les colonnes indésirables
df = df.drop(columns=['Poster_Link', 'Certificate', 'Overview'])


# Conversion des donnée “Runtime” en int
df["Runtime"] = df['Runtime'].str.extract('(\d+)').astype(int)

# Conversion des donnée “Gross” en int
df['Gross'] = df['Gross'].str.replace(',', '')

# Garder les donnée quantitative
df_quant = df.drop(columns=['Series_Title', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4'])
df_quant.to_csv('imdb_top_quant.csv', index=False)

# Garder les donnée quanlitative
df_qual = df.drop(columns=['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross'])
df_qual.to_csv('imdb_top_qual.csv', index=False)

df.to_csv('imdb_top.csv', index=False)


print(df_quant.head())


#Normalisation des données

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_quant)

# ACP

pca = PCA(n_components=3)
pca_res = pca.transform(x_scaled)

print("pca1",pca.explained_variance_ratio_)
print("pca_res",pca_res)

