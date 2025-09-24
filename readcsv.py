import pandas as pd

# Read the CSV file
df = pd.read_csv('imdb_top_1000.csv')

# Drop unwanted columns
df = df.drop(columns=['Poster_Link', 'Certificate', 'Overview'])

df.to_csv('imdb_top.csv', index=False)

print(df.head())
