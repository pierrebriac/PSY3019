import csv
import pandas as pd

from info_part import info_parti, enregistrer

# On lance le programme et on demande les données du participant
nom_entry, age_entry, sexe_entry, id_entry = info_parti()

# Récupérer les informations saisies par l'utilisateur
nom = nom_entry.get()
age = age_entry.get()
sexe = sexe_entry.get()
id_participant = id_entry.get()


from Paradigme_stroop import stroop

# On lance la tâche de Stroop
answer, couleurs, liste_mots, temps_reponse, iteration = stroop(10)

from Enregistre_answer import enregistrer_donnees

# On enregistre les données de la tâche de Stroop
enregistrer_donnees(answer, couleurs, liste_mots, temps_reponse, nom_entry, sexe_entry, age_entry, id_entry, iteration)

from Analyse_donnees import val_manquantes, recodage, afficher_resultats, score, statistiques

statistiques(id_entry)



