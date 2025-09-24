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

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# Supprimer les lignes avec "PG" dans toutes les colonnes
df = df[df.ne("PG").all(axis=1)]

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
pca_res = pca.fit_transform(x_scaled)

print(pca.explained_variance_ratio_) # Pourcentage de variance 

for i in range(len(df.index)):
    plt.text(pca_res[i,0]+0.2, pca_res[i,1], list(df.index)[i])
    plt.scatter(pca_res[i,0], pca_res[i,1], c='blue', edgecolors='k')
    
    
#Cercle de corrélation
pcs = pca.components_ #vecteurs propres
explained_var = pca.explained_variance_ratio_ #valeurs propres


# Coordonnees des variables (Correlations)

for i, col in enumerate(df_quant.columns):
    x=pcs[0,i]
    y=pcs[1,i]
    plt.arrow(0,0,x,y,head_width,head_length=0.03,fc='r',ec='r')
    plt.text(x*1.1,y*1.1,col,fontsize=9,color='r')
    
#cercle unité
theta = np.linspace(0, 2*np.pi, 300)
x = np.cos(theta) #abscisse
y = np.sin(theta) #ordonnés
plt.plot(x, y, "r--")
print(df.head())
