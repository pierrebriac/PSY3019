import pandas as pd
import numpy
import csv
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as create_table
import plotly.express as px
from geopy.geocoders import Nominatim
import seaborn as sns
import matplotlib.pyplot as plt

# Importer mon data frame
# Charger le fichier CSV existant
data = pd.read_csv("PierreBriacMétayerMariotti_données_psy3019-H23_20220211_distribution-human-rights-vdem.csv")


# On supprime toutes les données inférieurs à 1950 (les frontières des pays sont trop instables avant)
data.drop( data[ data['Year'] <= 1950 ].index, inplace=True) # Utilisation d'un comparateur
# On remplace "Yemen People's Republic" par "Yemen" pour simplifier
data=data.replace("Yemen People's Republic", "Yemen")


# On regarde les données manquantes
missing_values = data.isnull().sum()
print("Données manquantes")
print(missing_values)
# Enregistrer les missing_values dans un fichier CSV
missing_values.to_csv("missing_values.csv", index=True)


# Visualisation des valeurs manquantes à l'aide de la méthode heatmap de seaborn
fig = plt.figure() # pour sauvegarder
sns.heatmap(data.isnull(), cbar=False, cmap='plasma')
plt.show()
plt.close()
fig.savefig('missing_values.png')


# Regrouper les données autour des régions et par année
# On utilise ici le .mean() pour faire la moyenne de l'ensemble des pays d'un continent pour chaque année
df_grouped = data.groupby(['region', 'Year'], as_index=False).mean()


# On enregistre ce tableau en format csv
df_grouped.to_csv('data_grouped.csv', index=False)


# graphique du niveau de liberté dans chaque continent entre 1950 et 2021
fig = plt.figure() # pour sauvegarder
for region, group in df_grouped.groupby('region'):
    plt.plot(group['Year'], group['civ_libs_vdem_owid'], label=region)

plt.legend()
plt.xlabel('Années')
plt.ylabel('Libertés civiles')
plt.show()
plt.close()
fig.savefig('civil_liberty.png')


fig = plt.figure()
mean_by_region = df_grouped.groupby('region').mean()

# créer un histogramme avec la moyenne totale pour chaque région
mean_by_region['civ_libs_vdem_owid'].plot(kind='bar', x='region', 
                                          y='civ_libs_vdem_owid', 
                                          color = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'magenta'])

# Ajouter une étiquette d'axe Y
plt.ylabel('Score de libertés civiles')

# Afficher le graphique
plt.show()
plt.close()
fig.savefig('Liberte_mean_region.png')

# Nuage de points
mean_by_year = df_grouped.groupby('Year').mean() 
coefficients = np.polyfit(mean_by_year.index, mean_by_year['civ_libs_vdem_owid'], 1)
regression = np.poly1d(coefficients)
plt.scatter(mean_by_year.index, mean_by_year['civ_libs_vdem_owid'], color='red', s=5)
plt.plot(mean_by_year.index, regression(mean_by_year.index), color='magenta') #droite régression (x : années, y : moyenne des libertés)
# Ajouter une étiquette d'axe Y
plt.ylabel('Score de libertés civiles')
plt.xlabel('Années')
plt.show()

# Graphiques interactifs

plt.ion() # Activation de l'interactivité

i = 0
while i < 70:
    plt.plot(mean_by_year.index[:i], mean_by_year['civ_libs_vdem_owid'][:i])
    plt.xlabel('Année')
    plt.ylabel('Score de libertés civiles moyen')
    plt.title('Évolution de la liberté individuelle dans le monde')
    plt.get_current_fig_manager().full_screen_toggle()
    plt.pause(0.05) # Pause l'affichage pour 0.05 seconde
    plt.clf() # Efface le graphique précédent
    i += 1

plt.ioff() # Désactivation de l'interactivité
plt.show()

plt.ion() # Activation de l'interactivité
# Nuages de points interactifs
i = 0
while i < 70:
    plt.scatter(mean_by_year.index[:i], mean_by_year['civ_libs_vdem_owid'][:i],
                s=5, c='red')
    plt.xlabel('Année')
    plt.ylabel('Score de libertés civiles moyen')
    plt.title('Évolution de la liberté individuelle dans le monde')
    plt.get_current_fig_manager().full_screen_toggle()
    plt.pause(0.05) # Pause l'affichage pour 0.05 seconde
    plt.clf() # Efface le graphique précédent
    i += 1
plt.ioff() # Désactivation de l'interactivité
plt.show()

gapminder = px.data.gapminder() #charge les données de gapminder à partir de la bibliothèque plotly express

"""Les données gapminder sont un ensemble de données qui contiennent des informations sur les populations, 
les produits intérieurs bruts (PIB) et les taux de mortalité pour plusieurs pays sur plusieurs années."""

# Création d'un data frame contenant des pays et leur code iso_alpha qui permet de les localiser
df = pd.DataFrame({'country': gapminder.country, 'iso_alpha': gapminder.iso_alpha})
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Création d'une colonne localisation
data = data.assign(Localisation = pd.Series([np.nan] * len(data)))
data.reset_index(drop=True, inplace=True)

# Attribue à chaque pays de data une localisation lorsque celui ci est dans df
for i in range(len(data.Entity)) :
    pays = data.loc[i, "Entity"]
    if pays in df['country'].values:
        var = df.loc[df['country'] == pays]
        var.reset_index(drop=True, inplace=True)
        data.loc[i, "Localisation"] = var.loc[0, "iso_alpha"]
    else:
        continue

# Projette sur une carte le score des libertés civiles en fonction des années et des pays
graph_liberty = px.choropleth(data, locations = "Localisation", 
              color = "civ_libs_vdem_owid", hover_name = "Entity",
              animation_frame = "Year")

graph_liberty.update_layout(
    title="Libertés civiles dans le monde de 1950 à 2021",
    xaxis_title="Années",
    yaxis_title="Libertés civiles")

graph_liberty.write_html("Graphique_libertés_civiles.html")
graph_liberty.show()