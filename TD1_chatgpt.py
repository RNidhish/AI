import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== TD 1 : Analyse en Composantes Principales (ACP) - Version Améliorée ===\n")

# =============================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# =============================================================================
print("1. CHARGEMENT ET NETTOYAGE DES DONNÉES")
print("=" * 50)

# Lire le fichier CSV
print("Chargement du fichier imdb_top_1000.csv...")
df = pd.read_csv('imdb_top_1000.csv')
print(f"✓ Données chargées : {df.shape[0]} films, {df.shape[1]} colonnes")

# Afficher les informations de base
print(f"\nColonnes originales : {list(df.columns)}")
print(f"Valeurs manquantes par colonne :")
missing_data = df.isnull().sum()
for col in missing_data[missing_data > 0].index:
    print(f"  - {col}: {missing_data[col]} valeurs manquantes")

# Supprimer les colonnes indésirables
columns_to_drop = ['Poster_Link', 'Certificate', 'Overview']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_cols_to_drop:
    df = df.drop(columns=existing_cols_to_drop)
    print(f"✓ Colonnes supprimées : {existing_cols_to_drop}")

# Supprimer les lignes avec des valeurs manquantes
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"✓ Lignes avec valeurs manquantes supprimées : {rows_before - rows_after} lignes")

# Supprimer les lignes contenant "PG" (probablement dans la colonne Certificate)
rows_before = len(df)
df = df[df.ne("PG").all(axis=1)]
rows_after = len(df)
if rows_before != rows_after:
    print(f"✓ Lignes contenant 'PG' supprimées : {rows_before - rows_after} lignes")

print(f"\nDataset final après nettoyage : {df.shape[0]} films, {df.shape[1]} colonnes")

# =============================================================================
# 2. TRAITEMENT DES DONNÉES
# =============================================================================
print("\n2. TRAITEMENT DES DONNÉES")
print("=" * 30)

# Conversion des données "Runtime" en int
print("Conversion de la colonne Runtime...")
if 'Runtime' in df.columns:
    df_runtime_original = df['Runtime'].head(3)
    print(f"Exemples avant conversion : {df_runtime_original.tolist()}")
    df["Runtime"] = df['Runtime'].str.extract('(\d+)').astype(int)
    print(f"Exemples après conversion : {df['Runtime'].head(3).tolist()}")
    print(f"✓ Runtime convertie en entiers")

# Conversion des données "Gross" en float
print("\nConversion de la colonne Gross...")
if 'Gross' in df.columns:
    # Afficher quelques exemples avant conversion
    df_gross_original = df['Gross'].dropna().head(3)
    print(f"Exemples avant conversion : {df_gross_original.tolist()}")
    
    # Conversion avec gestion des virgules
    df['Gross'] = df['Gross'].astype(str).str.replace(',', '').replace('nan', np.nan)
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
    
    print(f"Exemples après conversion : {df['Gross'].dropna().head(3).tolist()}")
    print(f"✓ Gross convertie en float")

# Sauvegarder le dataset nettoyé
df.to_csv('imdb_top.csv', index=False)
print(f"✓ Dataset nettoyé sauvegardé : imdb_top.csv")

# =============================================================================
# 3. SÉPARATION DES DONNÉES QUANTITATIVES ET QUALITATIVES
# =============================================================================
print("\n3. SÉPARATION DES DONNÉES")
print("=" * 30)

# Définir explicitement les colonnes quantitatives et qualitatives
quant_cols = ['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']
qual_cols = ['Series_Title', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']

# Vérifier quelles colonnes existent dans le dataset
existing_quant_cols = [col for col in quant_cols if col in df.columns]
existing_qual_cols = [col for col in qual_cols if col in df.columns]

print(f"Colonnes quantitatives spécifiées : {quant_cols}")
print(f"Colonnes quantitatives trouvées : {existing_quant_cols}")
print(f"Colonnes qualitatives spécifiées : {qual_cols}")
print(f"Colonnes qualitatives trouvées : {existing_qual_cols}")

# Garder les données quantitatives
df_quant = df[existing_quant_cols]
print(f"\nDonnées quantitatives : {df_quant.shape}")
print(f"Variables : {list(df_quant.columns)}")

# Statistiques descriptives
print("\nStatistiques descriptives des données quantitatives :")
print(df_quant.describe().round(2))

df_quant.to_csv('imdb_top_quant.csv', index=False)
print("✓ Données quantitatives sauvegardées : imdb_top_quant.csv")

# Garder les données qualitatives
df_qual = df[existing_qual_cols]
print(f"\nDonnées qualitatives : {df_qual.shape}")
print(f"Variables : {list(df_qual.columns)}")

df_qual.to_csv('imdb_top_qual.csv', index=False)
print("✓ Données qualitatives sauvegardées : imdb_top_qual.csv")

# =============================================================================
# 4. ANALYSE EN COMPOSANTES PRINCIPALES (ACP)
# =============================================================================
print("\n4. ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
print("=" * 45)

# Vérification des données avant ACP
print("Vérifications préalables :")
print(f"- Nombre de variables quantitatives : {df_quant.shape[1]}")
print(f"- Nombre d'observations : {df_quant.shape[0]}")
print(f"- Valeurs manquantes : {df_quant.isnull().sum().sum()}")

# Matrice de corrélation
print(f"\nMatrice de corrélation :")
corr_matrix = df_quant.corr()
print(corr_matrix.round(3))

# Heatmap de la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Matrice de corrélation des variables quantitatives')
plt.tight_layout()
plt.show()

# Normalisation des données
print("\nNormalisation des données (StandardScaler)...")
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_quant)

print("Vérification de la normalisation :")
print(f"- Moyennes après normalisation : {np.mean(x_scaled, axis=0).round(6)}")
print(f"- Écarts-types après normalisation : {np.std(x_scaled, axis=0).round(6)}")

# ACP
print(f"\nApplication de l'ACP...")
n_components = min(5, df_quant.shape[1])  # Maximum 5 composantes ou le nombre de variables
pca = PCA(n_components=n_components)
pca_res = pca.fit_transform(x_scaled)

print(f"✓ ACP réalisée avec {n_components} composantes")

# Résultats de l'ACP
print("\nRésultats de l'ACP :")
print("Variance expliquée par composante :")
for i, var_exp in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var_exp:.4f} ({var_exp*100:.2f}%)")

print(f"\nVariance cumulée :")
cum_var = np.cumsum(pca.explained_variance_ratio_)
for i, cum in enumerate(cum_var):
    print(f"  PC1-PC{i+1}: {cum:.4f} ({cum*100:.2f}%)")

# =============================================================================
# 5. VISUALISATIONS
# =============================================================================
print("\n5. VISUALISATIONS")
print("=" * 20)

# Graphique des valeurs propres (Scree Plot)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, 
        alpha=0.7, color='skyblue', edgecolor='navy')
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, 
         marker='o', color='red', linewidth=2)
plt.xlabel('Composantes principales')
plt.ylabel('Variance expliquée')
plt.title('Scree Plot - Variance expliquée par composante')
plt.grid(True, alpha=0.3)

# Ajouter les pourcentages sur les barres
for i, v in enumerate(pca.explained_variance_ratio_):
    plt.text(i+1, v + 0.01, f'{v*100:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Projection des individus (films) - Version améliorée
plt.figure(figsize=(12, 9))

# Normaliser les valeurs de couleur pour éviter l'erreur
color_values = df_quant.iloc[:,0].astype(float)  # Convertir en float
color_values = (color_values - color_values.min()) / (color_values.max() - color_values.min())  # Normaliser entre 0 et 1

scatter = plt.scatter(pca_res[:,0], pca_res[:,1], 
                     c=color_values,  # Utiliser les valeurs normalisées
                     cmap='viridis', alpha=0.7, edgecolors='k', s=50)
plt.colorbar(scatter, label=f'{df_quant.columns[0]} (normalisé)')

# Ajouter quelques labels pour les films extrêmes
# Films les plus extrêmes sur PC1
extreme_pc1 = np.argsort(np.abs(pca_res[:,0]))[-5:]
# Films les plus extrêmes sur PC2
extreme_pc2 = np.argsort(np.abs(pca_res[:,1]))[-5:]
extreme_indices = np.union1d(extreme_pc1, extreme_pc2)

for i in extreme_indices:
    plt.annotate(f'Film {i}', 
                (pca_res[i,0], pca_res[i,1]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.title(f"Projection des films sur les deux premières composantes principales\n"
          f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}% - "
          f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}% "
          f"(Total: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.1f}%)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cercle des corrélations - Version améliorée
plt.figure(figsize=(10, 10))

# Calculer les coordonnées des variables
pcs = pca.components_
feature_names = df_quant.columns

# Dessiner les flèches et labels
for i, feature in enumerate(feature_names):
    x = pcs[0, i]
    y = pcs[1, i]
    
    # Flèche
    plt.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, 
             fc='red', ec='red', linewidth=2, alpha=0.8)
    
    # Label avec position intelligente
    offset = 0.15
    ha = 'left' if x > 0 else 'right'
    va = 'bottom' if y > 0 else 'top'
    
    plt.text(x + offset * np.sign(x), y + offset * np.sign(y), 
             feature, fontsize=10, color='darkred', 
             ha=ha, va=va, weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Cercle unité
theta = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(theta), np.sin(theta), "r--", alpha=0.7, linewidth=2)

# Grille et axes
plt.axhline(0, color='grey', lw=1, alpha=0.8)
plt.axvline(0, color='grey', lw=1, alpha=0.8)
plt.grid(True, alpha=0.3)

plt.title(f"Cercle des corrélations (PC1 vs PC2)\n"
          f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}% - "
          f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.axis("equal")
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.tight_layout()
plt.show()

# =============================================================================
# 6. INTERPRÉTATION DES RÉSULTATS
# =============================================================================
print("\n6. INTERPRÉTATION DES RÉSULTATS")
print("=" * 35)

print("Interprétation des composantes principales :")

# Analyse PC1
pc1_loadings = pcs[0, :]
pc1_sorted = sorted(zip(feature_names, pc1_loadings), key=lambda x: abs(x[1]), reverse=True)
print(f"\nPC1 ({pca.explained_variance_ratio_[0]*100:.1f}% de variance) :")
print("Variables les plus contributives :")
for var, loading in pc1_sorted[:3]:
    print(f"  - {var}: {loading:.3f}")

# Analyse PC2
if len(pcs) > 1:
    pc2_loadings = pcs[1, :]
    pc2_sorted = sorted(zip(feature_names, pc2_loadings), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nPC2 ({pca.explained_variance_ratio_[1]*100:.1f}% de variance) :")
    print("Variables les plus contributives :")
    for var, loading in pc2_sorted[:3]:
        print(f"  - {var}: {loading:.3f}")

# Qualité de la représentation
print(f"\nQualité de la représentation (2 premières composantes) :")
total_var_2pc = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
print(f"- Variance totale expliquée : {total_var_2pc*100:.1f}%")

if total_var_2pc > 0.7:
    print("✓ Excellente représentation (> 70%)")
elif total_var_2pc > 0.5:
    print("✓ Bonne représentation (> 50%)")
else:
    print("⚠ Représentation modérée (< 50%)")

# Affichage final du dataset
print(f"\n7. APERÇU FINAL DU DATASET")
print("=" * 30)
print(f"Dataset final : {df.shape}")
print(f"\nPremières lignes :")
print(df.head())

print(f"\nTypes de données :")
print(df.dtypes)

print("\n" + "="*80)
print("ANALYSE TERMINÉE - Fichiers générés :")
print("- imdb_top.csv (dataset nettoyé)")
print("- imdb_top_quant.csv (données quantitatives)")
print("- imdb_top_qual.csv (données qualitatives)")
print("="*80)