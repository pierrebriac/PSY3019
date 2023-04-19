import subprocess

# Lecture du fichier requirements.txt et stockage des noms de bibliothèques dans une liste
with open('requirements.txt') as f:
    libraries = [line.strip() for line in f]

# Vérification de la présence de chaque bibliothèque dans l'environnement Python
for library in libraries:
    try:
        __import__(library)
        print(f"{library} est déjà installée")
    except ImportError:
        print(f"{library} n'est pas installée. Installation en cours...")
        subprocess.check_call(["pip", "install", library])


# Importation des librairies
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Importation de mes fonctions
#from Data_analysis import DataAnalysis
from Fonctions_classes.fonction import create_sqlite_database, description, tri_insertion, knn_classification, svm_stats, pca_analysis, plot_variance_explained, plot_component_variance, codage_recursif

# Charger le fichier CSV existant
df = pd.read_csv("Donnees_libertes.csv")
description(df)

create_sqlite_database(df, 'Donnees/my_database.db')

from Fonctions_classes.Data_analysis import MyDataAnalyzer

print("*** Informations concernant la colonne Indice_liberte_expression ***")
analyzer = MyDataAnalyzer('Donnees/my_database.db')
analyzer.connect_to_db()
analyzer.load_data()
analyzer.check_missing_values()
analyzer.compute_statistics('Indice_liberte_expression')
analyzer.plot_histogram('Indice_liberte_expression')
analyzer.close_connection()

print("******************************************************")

# Trier la colonne Indice_moyen en utilisant l'algorithme de tri par insertion
tup_indice_moyen = tuple(df['Indice_moyen'].tolist())
sorted_tup = tri_insertion(tup_indice_moyen)

# Enregistrer la liste triée dans un fichier texte
np.savetxt("Donnees/sorted_indices.txt", sorted_tup, fmt='%.2f')

# Afficher un message de confirmation
print("La liste triée a été enregistrée dans le fichier 'sorted_indices.txt'.")
print("******************************************************")

# Select the features and target column names
features = df.columns[:-1]
target = 'Heureux_ou_non'

# Scale the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Create a new DataFrame with the scaled feature data and target variable
data_scaled = pd.DataFrame(X_scaled, columns=features)
data_scaled[target] = df[target]

# Use the knn_classification function to classify the data
accuracy = knn_classification(data_scaled, features, target)

print('Accuracy du KNN pour la colonne Heureux_ou_non :', accuracy)
print("******************************************************")
# Define a list of values to test for n_neighbors
n_neighbors_list = range(1, 21)

# Initialize an empty list to store the accuracy scores
accuracy_list = []

# Test each value of n_neighbors and record the accuracy score
for n_neighbors in n_neighbors_list:
    accuracy = knn_classification(data_scaled, features, target, n_neighbors=n_neighbors)
    accuracy_list.append(accuracy)

# Create a line plot of the accuracy scores
plt.plot(n_neighbors_list, accuracy_list)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Classification Accuracy')
plt.show()

# Save the plot to a file in the 'Donnees' directory
plt.savefig('Donnees/KNN_Classification_Accuracy.png')

svm_stats("Heureux_ou_non", ['Indice_liberte_expression', 'Indice_participation_electorale',
       'Indice_independance_medias', 'Indice_separation_pouvoirs',
       'Indice_lutte_corruption', 'Indice_protection_droits_minorites',
       'Indice_respect_liberte_presse', 'Indice_liberte_association',
       'Indice_liberte_reunion', 'Indice_liberte_religion',
       'Indice_acces_information', 'Indice_liberte_education',
       'Indice_justice_impartiale', 'Indice_elections_libres',
       'Indice_protection_droits_hommes', 'Indice_lutte_discrimination',
       'Indice_gouvernance_ouverte', 'Indice_protection_vie_privee',
       'Indice_liberte_internet', 'Indice_redevabilite_gouvernementale',
       'Indice_moyen', 'Indice_satisfaction', 'Niveau_stress',
       'Indicateur_sante_mentale', 'Niveau_activite_physique',
       'Indice_connexion_sociale'], df)

data_pca, pca = pca_analysis(df, n_components=20)
plot_variance_explained(pca)
plot_component_variance(pca)

# Code pour les deux fonctions anonymes

""" La fonction anonyme mean_liberte prend une ligne row de la banque de données en entrée et calcule la moyenne des cinq indices de liberté pertinents."""
mean_liberte = lambda row: (row.Indice_liberte_expression + row.Indice_liberte_association + row.Indice_liberte_reunion + row.Indice_liberte_religion + row.Indice_liberte_internet) / 5

"""
La fonction anonyme indice_bien_etre prend une ligne row de la banque de données en entrée et calcule 
l'indice de bien-être en combinant la moyenne des indices de bien-être, 
le niveau d'activité physique et l'indicateur de santé mentale."""

indice_bien_etre = lambda row: row.Moyenne_bien_etre * 0.6 + row.Niveau_activite_physique * 0.2 + row.Indicateur_sante_mentale * 0.2

# définition des fonctions anonymes
mean_liberte = lambda row: (row.Indice_liberte_expression + row.Indice_liberte_association + row.Indice_liberte_reunion + row.Indice_liberte_religion + row.Indice_liberte_internet) / 5
indice_bien_etre = lambda row: row.Moyenne_bien_etre * 0.6 + row.Niveau_activite_physique * 0.2 + row.Indicateur_sante_mentale * 0.2

# application des fonctions anonymes aux données
df['Moyenne_liberte'] = df.apply(mean_liberte, axis=1)
df['Indice_bien_etre'] = df.apply(indice_bien_etre, axis=1)

# création d'un tableau avec les deux listes
tableau = pd.DataFrame({'Moyenne_liberte': df['Moyenne_liberte'], 'Indice_bien_etre': df['Indice_bien_etre']})

# enregistrement du tableau dans le dossier "Donnees"
tableau.to_csv('Donnees/tableau_fcts_anonymes.csv', index=False)  # modifier 'tableau.csv' et 'Donnees' selon vos préférences
df = df.drop(['Moyenne_liberte', 'Indice_bien_etre'], axis=1)

df_encoded = codage_recursif(df)
df_encoded.to_csv('Donnees/df_encoded.csv', index=False)

