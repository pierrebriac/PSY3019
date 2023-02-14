
# Définir la fonction pour enregistrer les informations
def enregistrer(nom_entry, age_entry, sexe_entry, id_entry, root):

    import tkinter as tk
    import os

    nom = nom_entry.get()
    age = age_entry.get()
    sexe = sexe_entry.get()
    id_participant = id_entry.get()

    # Spécifier le nom du dossier et le chemin d'accès pour l'enregistrement du fichier individuel
    dossier = os.path.join(os.environ['HOME'], 'Desktop', 'Stroop_test', 'Info_participant')
    filename = os.path.join(dossier, "Info_participant_" + id_participant + ".csv")

    with open(filename, "w") as f:
        f.write(f"ID: {id_participant}\n")
        f.write(f"Nom: {nom}\n")
        f.write(f"Age: {age}\n")
        f.write(f"Sexe: {sexe}\n")
    
    root.quit()


def info_parti():

    import tkinter as tk

    # Initialiser la fenêtre principale
    root = tk.Tk()
    root.geometry("{}x{}".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    root.title("Demande d'informations")

    # Définir les étiquettes pour demander le nom, l'âge et le sexe
    nom_label = tk.Label(root, text="Quel est votre nom?", font=("Helvetica", 16))
    nom_label.pack()

    age_label = tk.Label(root, text="Quel est votre âge?", font=("Helvetica", 16))
    age_label.pack()

    sexe_label = tk.Label(root, text="Quel est votre sexe (M / F)?", font=("Helvetica", 16))
    sexe_label.pack()

    id_participant = tk.Label(root, text="Quel est votre ID ?", font=("Helvetica", 16))
    id_participant.pack()

    # Définir les entrées pour saisir les informations
    nom_entry = tk.Entry(root, font=("Helvetica", 16))
    nom_entry.pack()

    age_entry = tk.Entry(root, font=("Helvetica", 16))
    age_entry.pack()

    sexe_entry = tk.Entry(root, font=("Helvetica", 16))
    sexe_entry.pack()

    id_entry = tk.Entry(root, font=("Helvetica", 16))
    id_entry.pack()

    # Définir le bouton pour enregistrer les informations
    # Le mot-clé lambda crée une fonction anonyme qui permet de passer les arguments requis à la fonction enregistrer() lorsque le bouton est cliqué.
    bouton_enregistrer = tk.Button(root, text="Enregistrer", font=("Helvetica", 16), command=lambda: enregistrer(nom_entry, age_entry, sexe_entry, id_entry, root))

    bouton_enregistrer.pack()

    # Afficher la fenêtre principale
    root.mainloop()

    return nom_entry, age_entry, sexe_entry, id_entry

