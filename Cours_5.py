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

fig.savefig(os.path.join(chemin, 'my_figure.png'))