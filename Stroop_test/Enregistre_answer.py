def enregistrer_donnees(answer, couleurs, liste_mots, temps_reponse, nom, sexe_entry, age_entry, id_entry, iteration):
    import os
    import csv
    import pandas as pd

    # Récupération des données 
    age = age_entry.get()
    sexe = sexe_entry.get()
    id_participant = id_entry.get()


# Les données à ajouter au fichier global
    # Ouvrir le fichier en mode ajout

    df = pd.read_csv("Global.csv")

    donnees2 = {'ID':id_participant,
                    'Couleur du mot':couleurs,
                    'Mot':liste_mots,
                    'Reponse':answer,
                    'Temps de reponse':temps_reponse,
                    'Age':age,
                    'Genre': sexe,
                    'Iteration' : iteration}


    new_donnees = pd.DataFrame.from_dict(donnees2)
    concatenated_df = pd.concat([df, new_donnees])
    concatenated_df.to_csv('Global.csv',  index = False)

    # Spécifier le nom du dossier et le chemin d'accès pour l'enregistrement du fichier individuel
    dossier = os.path.join(os.environ['HOME'], 'Desktop', 'Stroop_test', 'Resultat_individuel')
    filename = os.path.join(dossier, "Donnees_participant_" + str(id_participant) + ".csv")

    # Créer un dictionnaire avec les listes et les noms de colonnes
    data = {'Couleurs': couleurs, 'Liste de mots': liste_mots, 'Reponses': answer, 'Temps de reponse':temps_reponse, 'Iteration' : iteration}

    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame(data)

    # Enregistrer le DataFrame en tant que fichier CSV
    df.to_csv(filename, index=False)




