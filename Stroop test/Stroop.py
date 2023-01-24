import pygame
import random
import pyautogui
import time
import csv

######################################### Questions utilisateurs #########################################
# Demander le nom de l'utilisateur
name = input("Quel est votre nom ? ")

# Demander l'âge de l'utilisateur
age = input("Quel est votre âge ? ")

# Demander le nom de l'utilisateur
genre = input("Quel est votre genre ? ")

# Demander l'âge de l'utilisateur
id_participant = input("ID du participant :  ")

# Enregistrer les informations dans un fichier
info_uti = "info_" + id_participant + ".txt"
with open(info_uti, "w") as file:
    file.write("Nom: " + name + "\n")
    file.write("Âge: " + age + "\n")
    file.write("Genre: " + genre + "\n")
    file.write("ID: " + id_participant + "\n")

######################################### Initialiser Pygame #########################################

# Initialiser Pygame
pygame.init()

# Initialiser la fenêtre
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Tâche de Stroop")
screen_resolution = pyautogui.size()

######################################### afficher les consignes sur l'écran #########################################
# afficher les consignes sur l'écran
pygame.display.set_caption("Consignes de l'expérience")
font = pygame.font.Font(None, 36)
text1 = font.render("Consignes de l'expérience : lorsque qu'un mot s'affiche il faut", True, (255, 255, 255))
text2 = font.render("appuyer sur la touche R (rouge), G(vert) ou B(bleu) en fonction de la couleur", True, (255, 255, 255))
text3 = font.render("indiqué par le mot (exemple red alors on appuie sur r) ", True, (255, 255, 255))
text4 = font.render("Appuyez sur la touche espace pour continuer", True, (255, 255, 255))


text_rect1 = text1.get_rect()
text_rect1.center = (screen_resolution[0] // 2, screen_resolution[1] // 3.5)
text_rect2 = text2.get_rect()
text_rect2.center = (screen_resolution[0] // 2, screen_resolution[1] // 3)
text_rect3 = text3.get_rect()
text_rect3.center = (screen_resolution[0] // 2, screen_resolution[1] // 2.5)
text_rect4 = text4.get_rect()
text_rect4.center = (screen_resolution[0] // 2, screen_resolution[1] // 1.5)

screen.blit(text1, text_rect1)
screen.blit(text2, text_rect2)
screen.blit(text3, text_rect3)
screen.blit(text4, text_rect4)

pygame.display.flip()

# attendre que l'utilisateur appuie sur la touche espace
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                waiting = False

######################################### Initialiser les données #########################################
# Définir les couleurs
colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

# Initialiser les variables pour stocker la réponse et le résultat
answer = []
couleurs = []
liste_mots = []
result = ""

# Initialiser le compteur de réponses
bonnes_reponses = 0
reponses = 0

# Démarrer le chronomètre
start_time = time.perf_counter()

######################################### Expérience #########################################
# Boucle principale
running = True
while running:
    if reponses >= 10 :
        break
    # Sélectionner des mots aléatoires et leur couleur associée
    words = ["red", "green", "blue"]
    word = random.choice(words)
    color = random.choice(list(colors.keys()))
    
    # Timer
    time = start_time

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.unicode == 'r':
                answer.append(event.unicode)
                couleurs.append(color)
                liste_mots.append(word)
            elif event.unicode == 'g':
                answer.append(event.unicode)
                couleurs.append(color)
                liste_mots.append(word)
            elif event.unicode == 'b':
                answer.append(event.unicode)
                couleurs.append(color)
                liste_mots.append(word)
                
            if answer[0] == color:
                result = "Correct!"
                bonnes_reponses += 1
                reponses += 1
                #waiting = False

            else:
                result = "Incorrect."
                reponses += 1
                #waiting = False

    screen.fill((255, 255, 255))

    # Afficher le mot
    font = pygame.font.Font(None, 150)
    text = font.render(word, True, colors[color])
    text_rect = text.get_rect(center=(screen_resolution[0]/2, screen_resolution[1]/2))
    screen.blit(text, text_rect)

    # Afficher la réponse
    font = pygame.font.Font(None, 60)
    text = font.render(result, True, (0, 0, 0))
    text_rect = text.get_rect(center=(screen_resolution[0]/2, screen_resolution[1]/1.8))
    screen.blit(text, text_rect)
    pygame.display.update()

# Ouvrir un fichier en écriture

filename = "data_" + id_participant + ".csv"

with open(filename, 'w', newline='') as file:
    # Créer un objet csv.writer
    writer = csv.writer(file)
    # Ecrire les données dans le fichier
    writer.writerow(['Couleurs'] + couleurs)
    writer.writerow(['Liste de mots'] + liste_mots)
    writer.writerow(['Réponses'] + answer)



# Quitter Pygame
pygame.quit()


