import pandas as pd

# Lire le fichier CSV
df = pd.read_csv('imdb_top_1000.csv')

# Supprimer les colonnes indésirables
df = df.drop(columns=['Poster_Link', 'Certificate', 'Overview'])

# Conversion des donnée “Runtime” en int
df["Runtime"] = df['Runtime'].str.extract('(\d+)').astype(int)

df.to_csv('imdb_top.csv', index=False)

print(df.head())