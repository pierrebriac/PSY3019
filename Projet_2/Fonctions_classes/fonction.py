# Importation des bibliothèques nécessaires 
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import pickle

def create_sqlite_database(df, db_name):
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql('my_table', conn, if_exists='replace', index=False)
        print(f"Database '{db_name}' created successfully!")
    except Exception as e:
        print(f"Error creating database: {str(e)}")
    finally:
        conn.close()

def description(df):
    # Statistiques descriptives
    describe = df.describe()
    # Enregistrer le tableau dans un fichier CSV
    describe.to_csv("Donnees/fichier_statistiques.csv")

def tri_insertion(tup):
    liste = list(tup) # le tuple permet d'économiser des ressources
    for i in range(1, len(liste)):
        valeur_courante = liste[i]
        position = i

        while position > 0 and liste[position - 1] > valeur_courante:
            liste[position] = liste[position - 1]
            position = position - 1

        liste[position] = valeur_courante

    return tuple(liste)

def knn_classification(data, features, target, test_size=0.33, random_state=42, n_neighbors=3):
    """
    Perform K-Nearest Neighbors classification on the given data using the specified features and target variable.
    
    Arguments:
    data -- pandas DataFrame containing the data
    features -- list of feature column names to use for classification
    target -- target column name to predict
    test_size -- proportion of the data to use for testing (default 0.33)
    random_state -- random seed for reproducibility (default 42)
    n_neighbors -- number of neighbors to consider (default 3)
    
    Returns:
    accuracy -- the accuracy score of the KNN classifier on the test data
    """
    # Create the design matrix X and target vector y
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Instantiate the KNN model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    pred = model.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, pred)

    return accuracy

def svm_stats(y_col, x_cols, df):
    # Sélectionner les colonnes d'intérêt pour les données d'entrée et de sortie
    X = df[x_cols]
    Y = df[y_col]
    
    # Encoder les étiquettes de sortie en utilisant LabelEncoder
    enc = LabelEncoder()
    label_encoder = enc.fit(Y)
    Y = label_encoder.transform(Y) + 1

    # Créer un classifieur SVM avec un noyau RBF
    clf = SVC(kernel='rbf', C=1E6)

    # Effectuer une normalisation ou une standardisation des données si nécessaire
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Effectuer une validation croisée à 5 plis
    scores = cross_val_score(clf, X, Y, cv=5)

    # Calculer la précision moyenne sur les 5 plis
    mean_score = scores.mean()

    # Afficher la précision moyenne sur les 5 plis
    print("Précision moyenne du modèle SVM : %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))

    # Entraîner le classifieur SVM sur toutes les données
    clf.fit(X, Y)

    # Créer un graphique illustrant la séparation de classes par le SVM
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel(x_cols[0])
    plt.ylabel(x_cols[1])
    plt.title("SVM pour la colonne " + y_col)
    plt.savefig("Donnees/" + y_col + "_SVM.png")
    
    # Enregistrer les données dans un fichier
    # code pour le formatage des chaînes est utilisé
    # code pour archivage de données
    with open("Donnees/SVM_stats.txt", "a") as f:
        f.write("Stats pour la colonne " + y_col + "\n")
        f.write("Précision moyenne: %0.2f (+/- %0.2f)\n" % (mean_score, scores.std() * 2))
        f.write("Vecteurs de support: %s\n" % clf.support_vectors_)
        f.write("Indices des vecteurs de support: %s\n" % clf.support_)
        f.write("Nombre de vecteurs de support pour chaque classe: %s\n\n" % clf.n_support_)

def pca_analysis(data, n_components):
    # Centrage et réduction des données
    data_scaled = (data - data.mean()) / data.std()

    # Création d'un objet PCA et ajustement des données
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)

    # Projection des données sur les nouvelles dimensions
    data_pca = pca.transform(data_scaled)

    return data_pca, pca

def plot_variance_explained(pca):
    # Calcul du pourcentage de variance expliquée pour chaque composante
    variance = pca.explained_variance_ratio_
    variance_cumulative = np.cumsum(variance)

    # Affichage du graphique
    fig, ax = plt.subplots()
    ax.plot(range(1, len(variance) + 1), variance_cumulative, 'o-')
    ax.set_xlabel('Nombre de composantes')
    ax.set_ylabel('% variance expliquée')
    ax.set_title('Variance expliquée par les composantes principales')
    ax.grid(True)

    # Enregistrement des données dans un fichier CSV
    data = {'composante': range(1, len(variance) + 1),
            'variance': variance,
            'variance_cumulative': variance_cumulative}
    df = pd.DataFrame(data)
    df.to_csv('Donnees/variance_pca.csv', index=False)

    # Enregistrement du graphique dans un fichier PNG
    fig.savefig('Donnees/variance_pca_cumule.png')

    # Affichage du graphique
    plt.show()

def plot_component_variance(pca):
    # Récupération du pourcentage de variance expliquée par chaque composante principale
    variance = pca.explained_variance_ratio_

    # Création du diagramme à barres
    fig, ax = plt.subplots()
    ax.bar(range(1, len(variance) + 1), variance)
    ax.set_xlabel('Composante principale')
    ax.set_ylabel('% variance expliquée')
    ax.set_title('Variance expliquée par chaque composante principale')
    ax.grid(True)

    # Enregistrement du graphique dans un fichier PNG
    fig.savefig('Donnees/variance_pca.png')

    # Affichage du diagramme à barres
    plt.show()

# La fonction codage_recursif prend un DataFrame pandas en entrée et retourne un nouveau DataFrame avec les valeurs numériques encodées récursivement.
def codage_recursif(df: pd.DataFrame) -> pd.DataFrame:

    # La fonction interne recursif convertit un nombre en base 10 (x) en sa représentation binaire récursive sous forme de chaîne de caractères.
    def recursif(x: float) -> str:
        # Si x est 0, retourner '0'.
        if x == 0:
            return '0'
        # Si x est 1, retourner '1'.
        elif x == 1:
            return '1'
        # Pour les autres valeurs de x, diviser x par 2 et calculer le quotient et le reste.
        else:
            quotient = int(x / 2)
            reste = int(x % 2)
            # Récursivement, appliquer la fonction recursif au quotient et concaténer le reste à la chaîne résultante.
            return recursif(quotient) + str(reste)

    # Créer une copie du DataFrame original pour ne pas le modifier directement.
    df_encoded = df.copy()
    
    # Parcourir toutes les colonnes du DataFrame.
    for col in df_encoded.columns:
        # Si le type de la colonne est 'float64' ou 'int64', appliquer la fonction recursif à chaque valeur de la colonne.
        if df_encoded[col].dtype == 'float64' or df_encoded[col].dtype == 'int64':
            df_encoded[col] = df_encoded[col].apply(recursif)
    
    # Retourner le DataFrame modifié avec les valeurs numériques encodées récursivement.
    return df_encoded
