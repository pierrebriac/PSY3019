import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt

chemin = os.path.join(os.environ['HOME'])

# Créer des données:

print(f'numpy version is: {np.__version__}')

dat = np.array([1, 2, 4, 8, 16, 32])

plt.style.use('classic')
# Comment enregistrer la figure?
fig = plt.figure() # pour sauvegarder, il faut créer une figure

plt.plot(dat)
plt.show()
plt.close()

fig.savefig(os.path.join(chemin, 'my_figure.png'))

# style MATLAB:
x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
plt.close()

# changeons le style de tracé
plt.style.use('seaborn-whitegrid')

plt.plot(dat, alpha = 0.2)
plt.show()
plt.close()

plt.plot(x, np.sin(x), '-', color='chartreuse')
plt.plot(x, np.cos(x), '--', color='magenta')

plt.show()
plt.close()

t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
plt.close()


# Inférence de style
# voici un exemple de tracé statique de style MATLAB:

# créer deux panneux et définir l'axe:

plt.subplot(2, 1, 1) # (lignes, colonnes, numéro de panneau)
plt.plot(x, np.sin(x))

# créer le deuxième panneau et définir l'axe:

plt.subplot (2,1,2)
plt.plot(x, np.cos(x))
plt.show()
plt.close()

data = np.random.randn(100)
plt.hist(data)
plt.show()
plt.close()


# histogrammes 2D

mean = [0,0]
cov = [[1,1],[1,2]]
x,y = np.random.multivariate_normal(mean, cov, 10000).T

plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.show()
plt.close()

# les histogrammes peut être hexagonal:

plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')

plt.show()
plt.close()

# les marqueurs peuvent également prendre différentes spécifications:

plt.plot(x, y, '-p', color='gray',
        markersize=15, linewidth=4,
        markerfacecolor='white', markeredgecolor='gray',
        markeredgewidth=2)
plt.ylim(-1.2, 1.2)

plt.show()
plt.close()

# EXEMPLE:

rng = np.random.RandomState(0) # méthode pour maintenir le même motif aléatoire

x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000*rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
# cmap est un argument pour spécifier une carte de couleur

# Pour afficher l'échelle de couleur - il faut utiliser le colorbar()
# la taille est donnée en pixels

plt.colorbar()
plt.show()
plt.close()


# EXEMPLE d'ensemble de données:

from sklearn.datasets import load_iris

iris=load_iris()

trait = iris.data.T
plt.scatter(trait[0], trait[1], alpha=0.2,
           s=100*trait[3], c=iris.target, cmap='viridis')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.close()

# pyplot.scatter peut également être utilisé pour tracer des graphiques polaires (polar plot)

# https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/polar_scatter.html

# base de données:
N = 150
r = 2 * np.random.rand(N)
area = 200 * r**2
theta = 2 * np.pi * np.random.rand(N)
colors = theta

fig = plt.figure()
ax  = fig.add_subplot(111, projection='polar')
c   = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
plt.show()
plt.close()

# sur un secteur:

fig = plt.figure()
ax  = fig.add_subplot(111, polar=True)
c   = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

ax.set_thetamin(45)
ax.set_thetamax(135)

plt.show()
plt.close()


x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy* np.random.randn(50)

# les barres d'erreur peuvent être personnalisées:

plt.errorbar(x, y, yerr=dy, fmt='o', color='chartreuse', ecolor='lightgray',
            elinewidth=3)#, capsize=0)
plt.show()
plt.close()
plt.errorbar(x, y, xerr=dy, fmt='o', color='chartreuse', ecolor='lightgray',
            elinewidth=3, capsize=0)
plt.show()
plt.close()

fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)

ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10,4),
           arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('local minimum', xy=(5* np.pi, -1), xytext=(2, -6),
           arrowprops=dict(arrowstyle="->",
                           connectionstyle='angle3,angleA=0,angleB=-90'))

plt.show()
plt.close()

def f(x,y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

# le tracé de contour prend 3 arguments: une grille de valeur x, valeur y
# et valuer z pour les niveaux de contour

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X,Y = np.meshgrid(x,y) # meshgrid construit des grilles 2D à partir de tableaux 1D
Z = f(X,Y)

plt.contour(X, Y, Z, colors='black')
plt.contour(X, Y, Z, 20, cmap='inferno')

# 20 spécifie que 20 lignes équidistantes doivent être tracées
plt.colorbar()

# RdGy représente la palette Red-Gray / Rouge-Gris - un bon choix pour les données centrées
plt.show()
plt.close()


# plt.imshow interprète une grille 2D comme une image

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='jet')
plt.colorbar()
plt.axis()#aspect='image')

# imshow() a quelques problèmes: n'accepte pas x et grid, donc ils doivent
# être spécifié manuellement sous une forme d'extension [xmin, xmax, ymin, ymax]
# l'origine de imshow() est à gauche en haut et non en bas.
plt.show()
plt.close()

# les tracés de contour et d'image peuvent être combinés:

contours = plt.contour(X, Y, Z, 3, colors='k')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5],
           origin='lower', cmap='hsv', alpha=0.5)
plt.colorbar()
plt.show()
plt.close()


