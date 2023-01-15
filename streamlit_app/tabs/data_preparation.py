import streamlit as st


title = "Préparation des données"
sidebar_name = "Préparation des données"


def run():

    st.title(title)

    st.markdown(
        """
        Point de départ : données table Results
        
        ## Classe du circuit

        Nous avons donc choisi de classer les circuits en 4 classes de vitesse. La méthode est la suivante : 
        - On possède les vitesses des tours les plus rapides (« fastestLapSpeed») de chaque Grand Prix (« RaceId »).
        - On prend la « fastestLapSpeed » la plus élevée
        - On regroupe les Grand Prix par circuit :
            - Le circuit d’Abu Dhabi a par exemple 14 « RaceId » différents car il y a eu 14 Grand Prix mais un seul « Circuit ID »
        - On fait la moyenne des « fastesLapSpeed » pour chaque circuit.
        - On découpe ces vitesses en 4 quartiles et on attribue à chaque circuit sa classe correspondante classée de 1 à 4.

        (image/dataframe)

        On se retrouve donc avec 4 classes que l’on pourrait définir par :
        - Circuit rapide
        - Circuit moyennement rapide
        - Circuit moyennement lent
        - Circuit lent

        ---

        ## Données météo

        La météo a toujours eu un impact en formule 1. Pour de nombreuses raisons, certains pilotes sont plus performants que d’autres sous la pluie, certaines écuries sont plus stratèges que d’autres et gèrent bien mieux les aléas climatiques, il y a plus d’accidents lors de gros épisodes de pluie ce qui peut bouleverser le classement…

        Pour ajouter les données météo on a utilisé le script présent ici :
        __[https://towardsdatascience.com/formula-1-race-predictor-5d4bfae887da](https://towardsdatascience.com/formula-1-race-predictor-5d4bfae887da)__

        L’idée du script est la suivante, pour chaque course, on a un lien Wikipédia. Sur la page Wikipédia, on trouve alors la météo de la course. Il « suffit » alors de compiler ces informations dans un dataframe.

        A l’aide d’un dictionnaire, on vient alors réunir des mots comme « nuages, nuageux, gris, sombre » dans une catégorie « nuageux ». Néanmoins, certaines pages ne présentaient pas l’information correctement ou avec des mots pas présents dans ce dictionnaire. Nous avons dû compléter à la main certaines informations (environ 3% des entrées) pour lesquelles il n’y avait aucune info remontée.

        La météo découpée en 5 attributs
        - Faisait-il chaud ?
        - Faisait-il froid ?
        - La piste était-elle sèche ?
        - La piste était-elle mouillée ?
        - Faisait-il nuageux ?

        (image/dataframe)

        ---

        ## Données classement et points

        Pour compléter notre jeu de données, on rajoute les données liées aux classements des pilotes et constructeurs au moment du départ du Grand Prix.

        Pour cela, on récupère ces données dans les datasets « Driver standing » et « Constructor standings ».

        Dans le jeu de données, on utilise une colonne intermédiaire « prev-raceId » correspondant à l’ID de la course précédente. On fusionne les données classements pilotes/écuries avec cette colonne comme clé.

        Nous devons effectuer un ajustement sur les 1ers Grand Prix de chaque saison. En effet, les données fusionnées sont celles du dernier Grand Prix de la saison précédentes. Il faut donc les réinitialiser à 0 pour avoir des données cohérentes.

        (image/dataframe)

        ---

        ## Traitement

        Une normalisation des données est appliquée avec la méthode **StandardScaler**

        """
    )