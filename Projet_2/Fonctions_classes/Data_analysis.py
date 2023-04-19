import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

class MyDataAnalyzer:
    
    def __init__(self, db_path):
        # Constructeur qui prend en entrée le chemin vers la base de données
        # et initialise les attributs conn et df à None
        self.db_path = db_path
        self.conn = None
        self.df = None
    
    def connect_to_db(self):
        # Méthode pour se connecter à la base de données
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("Connexion à la base de données réussie!")
        except Exception as e:
            print(f"Erreur lors de la connexion à la base de données: {str(e)}")
    
    def close_connection(self):
        # Méthode pour fermer la connexion à la base de données
        if self.conn:
            self.conn.close()
            print("Connexion fermée!")
    
    def load_data(self):
        # Méthode pour charger les données de la table dans un dataframe
        if not self.conn:
            print("Erreur: aucune connexion à la base de données.")
            return
        
        try:
            self.df = pd.read_sql_query("SELECT * FROM my_table", self.conn)
            print("Données chargées avec succès!")
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
    
    def check_missing_values(self):
        # Méthode pour vérifier la présence de valeurs manquantes dans les données
        if self.df is None:
            print("Erreur: aucune donnée chargée.")
            return
        
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("Attention: des valeurs manquantes ont été trouvées dans les colonnes suivantes:")
            print(missing_values[missing_values > 0])
        else:
            print("Aucune valeur manquante trouvée!")
    
    def compute_statistics(self, column_name):
        # Méthode pour calculer des statistiques descriptives pour une colonne spécifiée
        if self.df is None:
            print("Erreur: aucune donnée chargée.")
            return
        
        if column_name not in self.df.columns:
            print(f"Erreur: colonne '{column_name}' non trouvée dans les données.")
            return
        
        data = self.df[column_name].dropna()
        
        # Test de normalité Shapiro-Wilk
        w, p = stats.shapiro(data)
        text = f"Test de normalité Shapiro-Wilk: W = {w:.4f}, p-value = {p:.4f}\n"
        
        if p < 0.05:
            text += "Attention: les données ne semblent pas suivre une distribution normale!\n"
        
        # Calcul des statistiques descriptives
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        
        text += f"Moyenne: {mean:.4f}\n"
        text += f"Ecart-type: {std:.4f}\n"
        text += f"Médiane: {median:.4f}\n"
        text += f"Plage interquartile: {iqr:.4f}\n"
        
        # Enregistrer le texte dans un fichier texte
        if not os.path.exists("Donnees"):
            os.makedirs("Donnees")
        
        filename = f"Donnees/stats_{column_name}.txt"
        
        with open(filename, "w") as f:
            f.write(text)
        
        print(f"Les statistiques ont été enregistrées dans le fichier {filename}.")
    
    def plot_histogram(self, column_name):
        # Méthode pour tracer un histogramme pour une colonne spécifiée
        
        if self.df is None:
            print("Erreur: aucune donnée chargée.")
            return
        
        if column_name not in self.df.columns:
            print(f"Erreur: colonne '{column_name}' non trouvée dans les données.")
            return
        
        # Sélectionner les données de la colonne spécifiée et supprimer les valeurs manquantes
        data = self.df[column_name].dropna()
        
        # Créer une nouvelle figure et un nouvel axe
        fig, ax = plt.subplots()
        
        # Tracer l'histogramme avec les bins auto-sélectionnés
        ax.hist(data, bins='auto')
        
        # Ajouter des étiquettes pour l'axe x et l'axe y
        ax.set_xlabel(column_name)
        ax.set_ylabel("Fréquence")
        
        # Ajouter un titre à l'histogramme en utilisant le nom de la colonne spécifiée
        ax.set_title(f"Histogramme de {column_name}")
        
        # Vérifier si le dossier 'Donnees' existe, sinon le créer
        if not os.path.exists('Donnees'):
            os.makedirs('Donnees')
        
        # Enregistrer l'histogramme dans le dossier 'Donnees' avec le nom de la colonne spécifiée
        plt.savefig(f"Donnees/{column_name}_histogramme.png")
        
        # Afficher un message de confirmation
        print(f"L'histogramme de '{column_name}' a été enregistré dans le dossier 'Donnees'.")






