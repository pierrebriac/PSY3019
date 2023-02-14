def stroop(taille_exp):

    import pygame
    import time
    import random
    import numpy as np

    # Initialisation de Pygame
    pygame.init()

    # Création de la fenêtre d'affichage
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    # Liste des mots à afficher
    mots = ["Rouge", "Vert", "Bleu"]

    # Définir les couleurs
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

    # Initialiser les variables pour stocker la réponse et le résultat
    answer = []
    couleurs = []
    liste_mots = []
    temps_reponse = []
    iteration = np.arange(1,taille_exp + 1,1)
    nombre_mot = taille_exp
    # Boucle principale du programme
    for i in range(nombre_mot):
        # Effacement de l'écran
        screen.fill((255, 255, 255))

        # Affichage de la croix de fixation
        pygame.draw.line(screen, (0, 0, 0), (screen.get_width() // 2 - 10, screen.get_height() // 2), (screen.get_width() // 2 + 10, screen.get_height() // 2), 3)
        pygame.draw.line(screen, (0, 0, 0), (screen.get_width() // 2, screen.get_height() // 2 - 10), (screen.get_width() // 2, screen.get_height() // 2 + 10), 3)
        pygame.display.update()

        # Attendre 500 millisecondes
        pygame.time.wait(500)

        # Effacement de l'écran
        screen.fill((255, 255, 255))

        mot = random.choice(mots)
        color = random.choice(list(colors.keys()))

        # On ajoute dans les listes les données de l'expérience
        couleurs.append(color)
        liste_mots.append(mot)

        # Affichage du mot
        font = pygame.font.Font(None, 48)
        text = font.render(mot, True, colors[color])
        text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        screen.blit(text, text_rect)
        pygame.display.update()

        # Enregistrer le temps avant l'attente de la réponse de l'utilisateur
        debut_affichage = time.time()

        # Attente de la réponse de l'utilisateur
        reponse = None
        debut_attente = time.time()
        while reponse not in ["r", "v", "b"] and time.time() - debut_attente < 1:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.unicode in ["r", "v", "b"]:
                        reponse = event.unicode
                        if len(answer) < nombre_mot:
                            answer.append(event.unicode)

        # Enregistrer le temps de réponse de l'utilisateur
        temps_reponse.append(time.time() - debut_affichage)

        # Si l'utilisateur n'a pas répondu, ajouter "NA" à la liste answer
        if reponse is None:
            answer.append("NA")

        # Passage au mot suivant
        pygame.time.wait(500)  # Petit délai pour éviter les problèmes de clavier

    return answer, couleurs, liste_mots, temps_reponse, iteration




