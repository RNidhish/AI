import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Lire le fichier CSV
df = pd.read_csv('imdb_top_1000.csv')

# Supprimer les colonnes indésirables
df = df.drop(columns=['Poster_Link', 'Certificate', 'Overview'])

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# Supprimer les lignes contenant "PG"
df = df[df.ne("PG").all(axis=1)]

# Conversion des données “Runtime” en int
df["Runtime"] = df['Runtime'].str.extract('(\d+)').astype(int)

# Conversion des données “Gross” en int
df['Gross'] = df['Gross'].str.replace(',', '').astype(float)

# Garder les données quantitatives
df_quant = df.drop(columns=['Series_Title', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4'])

# Normalisation des données
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_quant)

# ACP
pca = PCA(n_components=3)
pca_res = pca.fit_transform(x_scaled)

print("Variance expliquée :", pca.explained_variance_ratio_)

# -----------------------------
# Projection des individus (films)
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(pca_res[:,0], pca_res[:,1], c='blue', edgecolors='k')

for i in range(len(df.index)):
    plt.text(pca_res[i,0]+0.1, pca_res[i,1], str(df.index[i]), fontsize=8)

plt.title("Projection des films sur les deux premières composantes principales")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# -----------------------------
# Cercle des corrélations
# -----------------------------
pcs = pca.components_
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(6,6))
for i, col in enumerate(df_quant.columns):
    x = pcs[0, i]
    y = pcs[1, i]
    plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='r', ec='r')
    plt.text(x*1.1, y*1.1, col, fontsize=9, color='r')

# Cercle unité
theta = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(theta), np.sin(theta), "r--")
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)

plt.title("Cercle des corrélations (PC1 vs PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axis("equal")
plt.show()

# Vérification du DataFrame
print(df.head())
