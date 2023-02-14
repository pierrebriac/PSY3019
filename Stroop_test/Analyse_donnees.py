def val_manquantes(data, glob_loc, id_participant):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import csv
    # On regarde les données manquantes
    missing_values = data.isnull().sum()

    # Visualisation des valeurs manquantes à l'aide de la méthode heatmap de seaborn
    fig = plt.figure() # pour sauvegarder
    sns.heatmap(data.isnull(), cbar=False, cmap='plasma')
    plt.show()
    plt.close()

    if glob_loc == "loc" :
        name = "Donnees_manquantes_global.png"
    if glob_loc == "glob" :
        dossier = os.path.join(os.environ['HOME'], 'Desktop', 'Stroop_test', 'Resultat_individuel_graph')
        name = os.path.join(dossier,"Donnees_manquantes_global_" + str(id_participant) + ".png")

    missing_values.to_csv(name, index=False)
    fig.savefig(name)
    

def recodage(data_participant, name, data_global):
    import os
    data_participant['Couleurs'] = data_participant['Couleurs'].replace({'blue': "Bleu", 'red': "Rouge", 'green': "Vert"})
    data_participant['Reponses'] = data_participant['Reponses'].replace({'b': "Bleu", 'r': "Rouge", 'v': "Vert"})
    dossier = os.path.join(os.environ['HOME'], 'Desktop', 'Stroop_test', 'Resultat_individuel')
    filename = os.path.join(dossier, name)
    data_participant.to_csv(filename, index=False)

    data_global['Couleur du mot'] = data_global['Couleur du mot'].replace({'blue': "Bleu", 'red': "Rouge", 'green': "Vert"})
    data_global['Reponse'] = data_global['Reponse'].replace({'b': "Bleu", 'r': "Rouge", 'v': "Vert"})
    data_global.to_csv('Global.csv', index=False)


def afficher_resultats(pourcentage_global, pourcentage_utilisateur, mean_groupe, mean_parti):
    import tkinter as tk
    
    # Créer la fenêtre
    fenetre = tk.Tk()
    fenetre.geometry("{}x{}".format(fenetre.winfo_screenwidth(), fenetre.winfo_screenheight()))
    fenetre.title("Résultats")

    # Ajouter l'étiquette pour le pourcentage de bonnes réponses globales
    etiquette_global = tk.Label(fenetre, text=f"Pourcentage de bonnes réponses globales : {pourcentage_global:.2f}%")
    etiquette_global.config(font=("Arial", 24))
    etiquette_global.pack(pady=20)

    # Ajouter l'étiquette pour le pourcentage de bonnes réponses globales
    etiquette_global_mean = tk.Label(fenetre, text=f"Temps moyen de réponse du groupe : {mean_groupe:.2f} secondes")
    etiquette_global_mean.config(font=("Arial", 24))
    etiquette_global_mean.pack(pady=20)

    # Ajouter l'étiquette pour le pourcentage de bonnes réponses de l'utilisateur
    etiquette_uti = tk.Label(fenetre, text=f"Pourcentage de bonnes réponses de l'utilisateur : {pourcentage_utilisateur:.2f}%")
    etiquette_uti.config(font=("Arial", 24))
    etiquette_uti.pack(pady=20)

    # Ajouter l'étiquette pour le pourcentage de bonnes réponses globales
    etiquette_uti_mean = tk.Label(fenetre, text=f"Temps moyen de réponse du groupe : {mean_parti:.2f} secondes")
    etiquette_uti_mean.config(font=("Arial", 24))
    etiquette_uti_mean.pack(pady=20)

    # Ajouter un bouton "Quitter"
    bouton_quitter = tk.Button(fenetre, text="Quitter", command=fenetre.quit)
    bouton_quitter.pack(pady=20)
    
    # Afficher la fenêtre
    fenetre.mainloop()
    fenetre.quit()


def score(data_participant, data_global):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import csv
    
    # Résultats globaux
    Liste_mot_glob = data_global["Mot"]
    Liste_rep_glob = data_global["Reponse"]

    # Résultats individuels
    Liste_mot_part = data_participant["Liste de mots"]
    Liste_rep_part = data_participant["Reponses"]

    # Compter le nombre de correspondances entre les deux listes
    nb_bonnes_reponses = 0
    for i in range(len(Liste_mot_glob)):
        if Liste_mot_glob[i] == Liste_rep_glob[i]:
            nb_bonnes_reponses += 1

    # Calculer le pourcentage de bonnes réponses
    pourcentage_bonnes_reponses_glob = nb_bonnes_reponses / len(Liste_mot_glob) * 100

    # Compter le nombre de correspondances entre les deux listes
    nb_bonnes_reponses_uti = 0
    for i in range(len(Liste_mot_part)):
        if Liste_mot_part[i] == Liste_rep_part[i]:
            nb_bonnes_reponses_uti += 1

    # Calculer le pourcentage de bonnes réponses
    pourcentage_bonnes_reponses_parti = nb_bonnes_reponses_uti / len(Liste_mot_part) * 100

    # Temps moyen de réponse du groupe
    mean_groupe = data_global["Temps de reponse"].mean()
    # Temps moyen de réponse du participant
    mean_parti = data_participant["Temps de reponse"].mean()
    afficher_resultats(pourcentage_bonnes_reponses_glob, pourcentage_bonnes_reponses_parti, mean_groupe, mean_parti)


def statistiques(id_entry):
    import pandas as pd
    import os
    import csv
    # Importer mon data frame global
    data_global = pd.read_csv("Global.csv")

    # Ouvrir data frame du participant
    id_participant = id_entry.get()
    #id_participant = 1

    dossier = os.path.join(os.environ['HOME'], 'Desktop', 'Stroop_test', 'Resultat_individuel')
    filename = os.path.join(dossier, "Donnees_participant_" + str(id_participant) + ".csv")
    name = "Donnees_participant_" + str(id_participant) + ".csv"

    data_participant = pd.read_csv(filename)
    recodage(data_participant, name, data_global)

    score(data_participant, data_global)

    val_manquantes(data_participant, "loc", id_participant)
    val_manquantes(data_global, "glob", id_participant)
