import streamlit as st
import pandas as pd

title = "Préparation des données"
sidebar_name = "Préparation des données"




def run():

    st.title(title)

    st.markdown(
        """

        La table « Results » est le point de départ de notre travail :
        
        """)
    results = pd.read_csv(r'..\data\results.csv',sep=',',index_col=0)
    st.dataframe(results.sample(10))   
    

    
    st.markdown(
        """
        ---

        ## Classe du circuit

        Nous avons donc choisi de classer les circuits en 4 classes de vitesse. La méthode est la suivante : 
        <ul><li>On possède les vitesses des tours les plus rapides (« <b>fastestLapSpeed</b> ») de chaque Grand Prix (« RaceId »).</li>
        <li>On prend la « <b>fastestLapSpeed</b> » la plus élevée</li>
        <li>On regroupe les Grands Prix par circuit.<br>
            <p>Par exemple :<br>
            Le circuit d’Abu Dhabi a 14 « <b>RaceId</b> » différents car il y a eu 14 Grands Prix mais un seul « <b>Circuit ID</b> »</p></li>
        <li>On fait la moyenne des « <b>fastesLapSpeed</b> » pour chaque circuit.</li>
        <li>On découpe ces vitesses en 4 quartiles et on attribue à chaque circuit sa classe correspondante classée de 1 à 4.</li></ul>
        
        
        """, unsafe_allow_html=True)

    st.image("assets/data_preparation_circuit_class.jpg")
    
    st.markdown(
          """  
          Ce qui nous donne la nouvelle colonne suivante dans le dataframe
          """)
    #import tableau final pour afficher la colonne fastestlapspeedclasses
    df_results_final = pd.read_csv(r'..\data\df_results_meteo_circuit_classement.csv',sep=';')
    st.dataframe(df_results_final['fastestLapSpeed_classes'].sample(10))
    
    
    st.markdown(
        """
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
        
        """)


    st.image("assets/data_preparation_weather.jpg")
        
    st.markdown(
        """
        ---

        ## Données classement et points

        Pour compléter notre jeu de données, on rajoute les données liées aux classements des pilotes et constructeurs au moment du départ du Grand Prix.

        Pour cela, on récupère ces données dans les datasets « __Driver standing__ » et « __Constructor standings__ ».

        Dans le jeu de données, on utilise une colonne intermédiaire « prev-raceId » correspondant à l’ID de la course précédente. On fusionne les données classements pilotes/écuries avec cette colonne comme clé.

        Nous devons effectuer un ajustement sur les 1ers Grand Prix de chaque saison. En effet, les données fusionnées sont celles du dernier Grand Prix de la saison précédentes. Il faut donc les réinitialiser à 0 pour avoir des données cohérentes.

        ---

        ## Dataframe final
        
        Après suppression des features inutiles, voici un échantillon du dataframe obtenu :

        """
        )

    df_results_final.dropna(inplace=True)
    st.dataframe(df_results_final.sample(10))
    
    # st.markdown(
    #     """
    #     ---
   
    #     ## Traitement

    #     Enfin, une normalisation des données est appliquée avec la méthode _**StandardScaler**_

    #     """
    # )