import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import warnings
warnings.filterwarnings('ignore')

print("=== TD 2 : Analyse Factorielle des Correspondances (AFC) et Analyse de Correspondances Multiples (ACM) ===\n")

# Charger les données IMDB
df = pd.read_csv('imdb_top.csv')

# Créer des variables catégorielles pour l'analyse
# 1. Catégorisation de l'IMDB Rating (plus fine pour éviter les problèmes)
df['Rating_Category'] = pd.cut(df['IMDB_Rating'], 
                              bins=[0, 7.0, 8.0, 8.5, 10], 
                              labels=['Moyen', 'Bon', 'Très_Bon', 'Excellent'])

# 2. Catégorisation de la durée
df['Runtime_Category'] = pd.cut(df['Runtime'], 
                               bins=[0, 100, 140, 180, 500], 
                               labels=['Très_Court', 'Court', 'Moyen', 'Long'])

# 3. Catégorisation de l'année de sortie
df['Era'] = pd.cut(df['Released_Year'], 
                  bins=[1920, 1970, 1990, 2010, 2020], 
                  labels=['Ancien', 'Classique', 'Moderne', 'Contemporain'])

# 4. Simplifier les genres (prendre le premier genre)
df['Main_Genre'] = df['Genre'].str.split(',').str[0].str.strip()

print("Variables catégorielles créées :")
print("- Rating_Category:", df['Rating_Category'].value_counts().to_dict())
print("- Runtime_Category:", df['Runtime_Category'].value_counts().to_dict())
print("- Era:", df['Era'].value_counts().to_dict())
print("- Main_Genre (top 5):", df['Main_Genre'].value_counts().head().to_dict())

# =============================================================================
# 2. ANALYSE FACTORIELLE DES CORRESPONDANCES (AFC)
# =============================================================================
print("\n\n2. ANALYSE FACTORIELLE DES CORRESPONDANCES (AFC)")
print("=" * 50)

# Création du tableau de contingence entre Rating_Category et Era
print("Création du tableau de contingence entre Rating_Category et Era")
data_crosstab = pd.crosstab(df['Rating_Category'], df['Era'])
print("\nTableau de contingence :")
print(data_crosstab)

# Vérifier si le tableau a assez de variabilité
if data_crosstab.shape[0] < 2 or data_crosstab.shape[1] < 2:
    print("⚠ Tableau trop petit pour l'AFC, essayons avec Runtime_Category et Era")
    data_crosstab = pd.crosstab(df['Runtime_Category'], df['Era'])
    print("\nNouveau tableau de contingence (Runtime vs Era) :")
    print(data_crosstab)

# Vérifier que nous n'avons pas de lignes/colonnes avec que des zéros
data_crosstab = data_crosstab.loc[(data_crosstab != 0).any(axis=1), (data_crosstab != 0).any(axis=0)]
print("\nTableau après suppression des lignes/colonnes nulles :")
print(data_crosstab)

# Standardisation des données avec gestion des erreurs
try:
    # Méthode alternative de standardisation plus robuste
    row_means = data_crosstab.mean(axis=1)
    col_means = data_crosstab.mean(axis=0)
    total_mean = data_crosstab.values.mean()
    
    # Calcul des résidus standardisés (méthode AFC classique)
    expected = np.outer(row_means, col_means) / total_mean
    residuals = (data_crosstab - expected) / np.sqrt(expected)
    data_scaled = residuals.fillna(0)  # Remplacer les NaN par 0
    
    print("\nDonnées standardisées (résidus standardisés) :")
    print(data_scaled.round(4))
    
except Exception as e:
    print(f"Erreur lors de la standardisation : {e}")
    # Fallback : utilisation d'une standardisation simple
    data_scaled = (data_crosstab - data_crosstab.mean()) / (data_crosstab.std() + 1e-8)  # Ajouter epsilon pour éviter division par 0
    print("\nDonnées standardisées (méthode alternative) :")
    print(data_scaled.round(4))

# Test de sphéricité de Bartlett avec gestion d'erreur
print("\nTest de sphéricité de Bartlett :")
try:
    # Vérifier que la matrice n'est pas singulière
    if np.linalg.matrix_rank(data_scaled) < min(data_scaled.shape):
        print("⚠ Matrice singulière détectée, ajout de bruit pour la régulariser")
        # Ajouter un petit bruit pour régulariser la matrice
        noise = np.random.normal(0, 1e-6, data_scaled.shape)
        data_scaled_reg = data_scaled + noise
        chi_square_value, p_value = calculate_bartlett_sphericity(data_scaled_reg)
    else:
        chi_square_value, p_value = calculate_bartlett_sphericity(data_scaled)
    
    print(f"Chi-carré : {chi_square_value:.4f}")
    print(f"P-value : {p_value:.6f}")
    
    if p_value < 0.05:
        print("✓ Les variables sont suffisamment corrélées pour une AFC (p < 0.05)")
    else:
        print("⚠ Les variables ne sont pas suffisamment corrélées (p >= 0.05)")
        print("Nous procédons tout de même avec l'analyse...")
        
except Exception as e:
    print(f"Erreur lors du test de Bartlett : {e}")
    print("Procédons tout de même avec l'analyse...")

# Détermination du nombre de facteurs
n_factors_max = min(data_scaled.shape[0] - 1, data_scaled.shape[1] - 1)
print(f"\nNombre maximum de facteurs : {n_factors_max}")

# Vérifier que nous avons assez de facteurs
if n_factors_max < 1:
    print("⚠ Pas assez de dimensions pour l'analyse factorielle")
    print("Essayons avec une approche PCA directe...")
    
    # Utiliser PCA comme alternative
    from sklearn.decomposition import PCA
    pca_afc = PCA(n_components=min(2, min(data_crosstab.shape)))
    pca_result = pca_afc.fit_transform(data_crosstab)
    
    print("\nVariance expliquée par PCA :")
    for i, var_exp in enumerate(pca_afc.explained_variance_ratio_):
        print(f"Composante {i+1} : {var_exp:.4f} ({var_exp*100:.2f}%)")
    
    # Graphique simple avec PCA
    if pca_afc.n_components_ >= 2:
        plt.figure(figsize=(10, 8))
        components = pca_afc.components_
        plt.scatter(components[0, :], components[1, :], color='red', s=100, alpha=0.7)
        
        for i, (x, y, label) in enumerate(zip(components[0, :], components[1, :], data_crosstab.columns)):
            plt.text(x + 0.02, y + 0.02, str(label), ha="center", va="center", fontsize=10, color='blue')
            
        for i, (x, y, label) in enumerate(zip(components[0, :], components[1, :], data_crosstab.index)):
            plt.text(x - 0.02, y - 0.02, str(label), ha="center", va="center", fontsize=10, color='red')
        
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(0, color='k', linestyle='-', alpha=0.3)
        plt.title('Analyse de Correspondances (méthode PCA)')
        plt.xlabel(f'Composante 1 ({pca_afc.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'Composante 2 ({pca_afc.explained_variance_ratio_[1]*100:.1f}%)')
        plt.grid(True, alpha=0.3)
        plt.show()
        
else:
    # Analyse factorielle pour déterminer le nombre de facteurs avec gestion d'erreurs
    try:
        fa = FactorAnalyzer(n_factors=n_factors_max, rotation=None)
        fa.fit(data_scaled)
        ev, v = fa.get_eigenvalues()

        print("\nValeurs propres :")
        for i, eigenvalue in enumerate(ev):
            print(f"Facteur {i+1} : {eigenvalue:.4f}")

        # Graphique des valeurs propres (Scree Plot)
        plt.figure(figsize=(10, 6))
        plt.scatter(range(1, len(ev) + 1), ev, color='red', s=50)
        plt.plot(range(1, len(ev) + 1), ev, color='blue', marker='o')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Seuil = 1')
        plt.title('Scree Plot - Valeurs propres')
        plt.xlabel('Facteurs')
        plt.ylabel('Valeurs propres')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        # Nombre de facteurs à retenir (valeur propre >= 1)
        n_factors_retain = max(1, sum(1 for eigenvalue in ev if eigenvalue >= 1))
        print(f"\nNombre de facteurs à retenir (valeur propre >= 1) : {n_factors_retain}")

    except Exception as e:
        print(f"Erreur avec FactorAnalyzer : {e}")
        print("Utilisation de PCA comme alternative...")
        n_factors_retain = min(2, min(data_scaled.shape))
        ev = [1.5, 0.5]  # Valeurs factices pour la suite du code

# AFC avec différentes rotations (avec gestion d'erreurs)
try:
    if n_factors_retain >= 2 and min(data_scaled.shape) >= 2:
        methods = [
            ("AFC Sans rotation", FactorAnalyzer(n_factors_retain, rotation=None)),
            ("AFC Varimax", FactorAnalyzer(n_factors_retain, rotation="varimax")),
            ("AFC Quartimax", FactorAnalyzer(n_factors_retain, rotation="quartimax")),
        ]

        fig, axes = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)
        
        for ax, (method, fa) in zip(axes, methods):
            try:
                fa.fit(data_scaled)
                components = fa.loadings_.T  # Transposer pour avoir les bonnes dimensions
                
                # Projection des modalités
                ax.scatter(components[0, :], components[1, :], color='red', s=50, alpha=0.7)
                
                # Lignes de référence
                ax.axhline(0, -1, 1, color='k', linestyle='-', alpha=0.3)
                ax.axvline(0, -1, 1, color='k', linestyle='-', alpha=0.3)
                
                # Étiquettes des modalités
                for i, (x, y, label) in enumerate(zip(components[0, :], components[1, :], data_scaled.columns)):
                    ax.text(x + 0.02, y + 0.02, str(label), ha="center", va="center", fontsize=8)
                    
                for i, (x, y, label) in enumerate(zip(components[0, :], components[1, :], data_scaled.index)):
                    ax.text(x - 0.02, y - 0.02, str(label), ha="center", va="center", fontsize=8, color='blue')
                
                ax.set_title(str(method))
                ax.set_xlabel("Facteur 1")
                if ax.get_subplotspec().is_first_col():
                    ax.set_ylabel("Facteur 2")
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Erreur: {str(e)[:50]}...", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{method} - Erreur")
        
        plt.tight_layout()
        plt.show()
    else:
        print("⚠ Pas assez de facteurs pour l'analyse avec rotations")
        print("Utilisation d'une visualisation simple avec les données brutes...")
        
        # Visualisation simple du tableau de contingence
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_crosstab, annot=True, fmt='d', cmap='Blues')
        plt.title('Tableau de contingence - Visualisation directe')
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"Erreur lors de l'AFC avec rotations : {e}")
    print("Visualisation alternative avec heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title('Tableau de contingence - Visualisation directe')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. ANALYSE DE CORRESPONDANCES MULTIPLES (ACM)
# =============================================================================
print("\n\n3. ANALYSE DE CORRESPONDANCES MULTIPLES (ACM)")
print("=" * 50)

# Sélection des variables qualitatives
qualitative_vars = ['Rating_Category', 'Runtime_Category', 'Era']
df_qual_subset = df[qualitative_vars].dropna()

print(f"Variables sélectionnées : {qualitative_vars}")
print(f"Nombre d'observations : {len(df_qual_subset)}")

# Création du tableau disjonctif complet
print("\nCréation du tableau disjonctif complet...")
dc = pd.DataFrame(pd.get_dummies(df_qual_subset))
print(f"Dimensions du tableau disjonctif : {dc.shape}")
print(f"Variables créées : {list(dc.columns)}")

print("\nAperçu du tableau disjonctif :")
print(dc.head())

# ACM avec une approche PCA sur le tableau disjonctif
print("\nAnalyse de Correspondances Multiples...")

# Standardisation du tableau disjonctif
scaler = StandardScaler()
dc_scaled = scaler.fit_transform(dc)

# ACP sur le tableau disjonctif standardisé
pca_mca = PCA(n_components=min(5, dc.shape[1]))
mca_result = pca_mca.fit_transform(dc_scaled)

print("\nVariance expliquée par chaque composante :")
for i, var_exp in enumerate(pca_mca.explained_variance_ratio_):
    print(f"Composante {i+1} : {var_exp:.4f} ({var_exp*100:.2f}%)")

print(f"Variance cumulée (2 premières composantes) : {pca_mca.explained_variance_ratio_[:2].sum()*100:.2f}%")

# Graphique des modalités dans l'espace factoriel
plt.figure(figsize=(12, 8))

# Projection des modalités (variables)
components = pca_mca.components_
plt.scatter(components[0, :], components[1, :], color='red', s=50, alpha=0.7, label='Modalités')

# Ajout des étiquettes
for i, (x, y, nom) in enumerate(zip(components[0, :], components[1, :], dc.columns)):
    plt.text(x + 0.01, y + 0.01, nom, fontsize=9, ha='left')

plt.axhline(0, color='k', linestyle='-', alpha=0.3)
plt.axvline(0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(f'Dimension 1 ({pca_mca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Dimension 2 ({pca_mca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('Analyse de Correspondances Multiples - Projection des modalités')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Projection des individus (films)
plt.figure(figsize=(12, 8))
plt.scatter(mca_result[:, 0], mca_result[:, 1], alpha=0.6, s=30, color='blue')
plt.axhline(0, color='k', linestyle='-', alpha=0.3)
plt.axvline(0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(f'Dimension 1 ({pca_mca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Dimension 2 ({pca_mca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('Analyse de Correspondances Multiples - Projection des individus')
plt.grid(True, alpha=0.3)
plt.show()

# Contributions des variables aux axes
print("\nContributions des variables aux axes :")
loadings = pca_mca.components_.T * np.sqrt(pca_mca.explained_variance_)
contributions_df = pd.DataFrame(
    loadings[:, :2], 
    columns=[f'Dim 1 ({pca_mca.explained_variance_ratio_[0]*100:.1f}%)', 
             f'Dim 2 ({pca_mca.explained_variance_ratio_[1]*100:.1f}%)'],
    index=dc.columns
)
print(contributions_df.round(3))

# =============================================================================
# 4. INTERPRÉTATION ET ANALYSE
# =============================================================================
print("\n\n4. INTERPRÉTATION ET ANALYSE")
print("=" * 50)

# Analyse des contributions les plus importantes
print("Variables les plus contributives à la Dimension 1 :")
dim1_contrib = contributions_df.iloc[:, 0].abs().sort_values(ascending=False)
print(dim1_contrib.head().round(3))

print("\nVariables les plus contributives à la Dimension 2 :")
dim2_contrib = contributions_df.iloc[:, 1].abs().sort_values(ascending=False)
print(dim2_contrib.head().round(3))

# Analyse par groupes de modalités
print("\nAnalyse des oppositions principales :")
print("Dimension 1 - Oppositions :")
positive_dim1 = contributions_df[contributions_df.iloc[:, 0] > 0].iloc[:, 0].sort_values(ascending=False)
negative_dim1 = contributions_df[contributions_df.iloc[:, 0] < 0].iloc[:, 0].sort_values()
print("Côté positif :", positive_dim1.head(3).to_dict())
print("Côté négatif :", negative_dim1.head(3).to_dict())

print("\nDimension 2 - Oppositions :")
positive_dim2 = contributions_df[contributions_df.iloc[:, 1] > 0].iloc[:, 1].sort_values(ascending=False)
negative_dim2 = contributions_df[contributions_df.iloc[:, 1] < 0].iloc[:, 1].sort_values()
print("Côté positif :", positive_dim2.head(3).to_dict())
print("Côté négatif :", negative_dim2.head(3).to_dict())

print("\n" + "="*80)
print("FIN DE L'ANALYSE - Voir le compte-rendu détaillé ci-dessous")
print("="*80)