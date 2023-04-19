# README
# Description
Ce code est une application de différents algorithmes de traitement de données pour une banque de données sur le bien-être dans différents pays. Les algorithmes incluent le tri par insertion, la classification KNN, la classification SVM, l'analyse en composantes principales (PCA), et la création de fonctions anonymes.

# Bibliothèques utilisées
Ce code nécessite l'utilisation des bibliothèques suivantes :

csv
pandas
matplotlib.pyplot
os
numpy
sklearn

Si l'installation automatique ne fonctionne pas, exécuter la commande suivante :

pip install -r requirements.txt

# Utilisation
Pour exécuter le code, il suffit de lancer le fichier "main.py" à partir de l'invite de commande ou de l'IDE.

Le code effectue les opérations suivantes :

Vérification de la présence de chaque bibliothèque dans l'environnement Python
Lecture d'un fichier CSV contenant les données sur le bien-être
Création d'une base de données SQLite
Analyse des données en utilisant la classe "MyDataAnalyzer"
Tri de la colonne "Indice_moyen" en utilisant l'algorithme de tri par insertion
Sauvegarde de la liste triée dans un fichier texte
Classification des données en utilisant l'algorithme KNN
Affichage de l'accuracy du KNN pour la colonne "Heureux_ou_non"
Création d'un graphique de l'accuracy du KNN en fonction du nombre de voisins
Classification des données en utilisant l'algorithme SVM
Analyse en composantes principales (PCA) des données
Création de deux fonctions anonymes pour calculer la moyenne des indices de liberté et l'indice de bien-être
Application des fonctions anonymes aux données
Enregistrement des données modifiées dans des fichiers CSV.
# Auteur
Ce code a été écrit par METAYER--MARIOTTI Pierre-Briac.