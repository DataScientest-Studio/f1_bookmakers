import streamlit as st
import pandas as pd

title = "Exploration des données"
sidebar_name = "Exploration des données"

def run():
    # Entête
    st.title(title)

    st.markdown(
        """
        ## Ergast

        Les données sont principalement fournies par le service web __[Ergast](http://ergast.com/mrd/)__ qui contient des données historiques de la Formule 1 depuis le début de la discipline en 1950.

        La structure de la base de données est présentée ci-dessous :
        """
    )

    st.image("http://ergast.com/images/ergast_db.png")

    # Liste tables
    st.markdown(
            """
            #### Liste des tables :
            - [Circuits](#circuits)
            - [Constructor results](#constructors-results)
            - [Constructor standings](#constructors-standings)
            - [Constructors](#constructors)
            - [Driver standings](#driver-standings)
            - [Drivers](#drivers)
            - [Lap times](#lap-times)
            - [Pit stops](#pit-stops)
            - [Qualifying](#qualifying)
            - [Races](#races)
            - [Results](#results)
            - [Seasons](#seasons)
            - [Sprint results](#sprint-results)
            - [Status](#status)
            """
        )

    st.markdown('---')

    # Table Circuits
    st.markdown(
        """
        ### Circuits

        La table « Circuits » fournit les informations des différents circuits sur lesquels se sont déroulés les Grands Prix dans l’histoire de la Formule 1.
        """
    )
    if st.checkbox('Voir la table « Circuits »'):
        st.markdown(
            """
            - __circuitId__ = ID (clé primaire)
            - __circuitRef__ = ID (texte)
            - __name__ = Nom du circuit
            - __location__ = Ville
            - __country__ = Pays
            - __lat__ = Latitude
            - __lng__ = Longitude
            - __alt__ = Altitude
            - __url__ = Page Wikipédia
            """
        )
        df = pd.read_csv(r'../data/circuits.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Constructors results
    st.markdown(
        """
        ### Constructors results

        La table « Constructor results » fournit les résultats des écuries à l’issue des Grands Prix.
        """
    )
    if st.checkbox('Voir la table « Constructor results »'):
        st.markdown(
            """
            - __constructorResultsId__ = ID (clé primaire)
            - __raceId__ = ID du Grand prix
            - __constructorId__ = ID écurie
            - __points__ = Points gagnés par l'écurie
            - __status__ = 'D' pour disqualifié ou valeur nulle
            """
        )
        
        df = pd.read_csv(r'../data/constructor_results.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Constructors standings
    st.markdown(
        """
        ### Constructors standings

        La table « Constructor standings » fournit le classement actualisé des écuries au Championnat Constructeurs à l’issue de chaque Grand Prix.
        """
    )
    if st.checkbox('Voir la table « Constructor standings »'):
        st.markdown(
            """
            - __constructorStandingsId__ = ID (clé primaire)
            - __raceId__ = ID du Grand prix
            - __constructorId__ = ID écurie
            - __points__ = Points gagnés par l'écurie et cumulés au fil des Grands Prix d'une saison
            - __position__ = Position écurie au classement constructeurs
            - __positionText__ = Idem 'position' au format texte
            - __wins__ = Nombre de victoires cumulées au fil d'une saison
            """
        )
        
        df = pd.read_csv(r'../data/constructor_standings.csv')

        st.dataframe(df.head())
    
    st.markdown('---')

    # Table Constructors
    st.markdown(
        """
        ### Constructors

        La table « Constructor » fournit les informations sur les différentes écuries ayant participé aux championnats dans l’histoire de la Formule 1.
        """
    )
    if st.checkbox('Voir la table « Constructor »'):
        st.markdown(
            """
            - __constructorId__ = ID (clé primaire)
            - __constructorRef__ = ID (texte)
            - __name__ = Nom écurie
            - __nationality__ = Nationalité
            - __url__ = Page Wikipédia
            """
        )
        
        df = pd.read_csv(r'../data/constructors.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Driver standings
    st.markdown(
        """
        ### Driver standings

        La table « Driver standings » fournit le classement actualisé des pilotes au Championnat à l’issue de chaque Grand Prix.
        """
    )
    if st.checkbox('Voir la table « Driver standings »'):
        st.markdown(
            """
            - __driverStandingsId__ = ID (clé primaire)
            - __raceId__ = ID du Grand prix
            - __driverId__ = ID du pilote
            - __points__ = Points gagnés par le pilote et cumulés au fil des Grands Prix d'une saison
            - __position__ = Position au classement pilotes
            - __positionText__ = Idem 'position' au format texte
            - __wins__ = Nombre de victoires cumulées au fil d'une saison
            """
        )
        
        df = pd.read_csv(r'../data/driver_standings.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Drivers
    st.markdown(
        """
        ### Drivers

        La table « Drivers » fournit les informations sur les différents pilotes ayant participé aux championnats dans l’histoire de la Formule 1.
        """
    )
    if st.checkbox('Voir la table « Drivers »'):
        st.markdown(
            """
            - __driverId__ = ID (clé primaire)
            - __driverRef__ = ID (texte)
            - __number__ = Numéro du pilote ou valeur nulle 
            - __code__ = Initiales du pilote
            - __forename__ = Prénom
            - __surname__ = Nom
            - __dob__ = date de naissance
            - __nationality__ = nationalité
            - __url__ = page Wikipédia
            """
        )
        
        df = pd.read_csv(r'../data/drivers.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Lap times
    st.markdown(
        """
        ### Lap times

        La table « Lap times » fournit les chronomètres par tour des pilotes à chaque Grand Prix.

        __Note__ : Les données sur les chronomètres sont disponibles à partir de la saison _1996_.
        """
    )
    if st.checkbox('Voir la table « Lap times »'):
        st.markdown(
            """
            - __raceId__ = ID du Grand Prix
            - __driverId__ = ID du pilote
            - __lap__ = Numéro du tour
            - __position__ = Positon du pilote sur le tour
            - __time__ = Chrono du pilote sur le tour
            - __milliseconds__ = Chrono converti en millisecondes
            """
        )
        
        df = pd.read_csv(r'../data/lap_times.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Pit stops
    st.markdown(
        """
        ### Pit stops

        La table « Pit stops » fournit les arrêts aux stands effectués par les pilotes à chaque Grand Prix.

        __Note__ : Les données sur les arrêts au stand sont disponibles à partir de la saison _2011_.
        """
    )
    if st.checkbox('Voir la table « Pit stops »'):
        st.markdown(
            """
            - __raceId__ = ID du Grand Prix
            - __driverId__ = ID du pilote
            - __stop__ = Numéro arrêt aux stands
            - __lap__ = Numéro du tour
            - __time__ = Temps début arrêt
            - __duration__ = Durée arrêt
            - __milliseconds__ = Durée convertie en millisecondes
            """
        )
        
        df = pd.read_csv(r'../data/pit_stops.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Qualifying
    st.markdown(
        """
        ### Qualifying

        La table « Qualifying » fournit les résultats des séances de qualifications des pilotes à chaque Grand Prix.
        
        __Note__ : Les résultats des qualifications ne sont entièrement pris en charge qu'à partir de la saison _2003_.

        Les positions sur la grille de départ peuvent être différentes de celles des qualifications, en raison de pénalités ou de problèmes mécaniques.
        """
    )
    if st.checkbox('Voir la table « Qualifying »'):
        st.markdown(
            """
            - __qualifyId__ = ID (clé primaire)
            - __raceId__ = ID du Grand Prix
            - __driverId__ = ID du pilote
            - __constructorId__ = ID écurie
            - __number__ = Numéro du pilote
            - __position__ = Position pour la grille de départ de la course
            - __q1__ = Chrono en première phase de qualification
            - __q2__ = Chrono en deuxième phase de qualification ou nul si disqualifié Q1
            - __q3__ = Chrono en troisème phase de qualification ou nul si disqualifié Q2
            """
        )
        
        df = pd.read_csv(r'../data/qualifying.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Races
    st.markdown(
        """
        ### Races

        La table « Races » fournit les informations sur les différents Grands Prix organisés dans l’histoire de la Formule 1.
        """
    )
    if st.checkbox('Voir la table « Races »'):
        st.markdown(
            """
            - __raceId__ = ID (clé primaire)
            - __year__ = Année
            - __round__ = Numéro du Grand Prix sur la saison
            - __circuitId__ = ID du circuit
            - __name__ = Nom du Grand prix
            - __date__ = Date du Grand prix
            - __time__ = Heure départ de la course
            - __url__ = Page Wikipédia
            - __fp1_date__ | __fp1_time__ = Date et heure départ des essais libres 1 (ou nul)
            - __fp2_date__ | __fp2_time__ = Date et heure départ des essais libres 2 (ou nul)
            - __fp3_date__ | __fp3_time__ = Date et heure départ des essais libres 3 (ou nul)
            - __quali_date__ | __quali_time__ = Date et heure départ de la séance de qualifications (ou nul)
            - __sprint_date__ | __sprint_time__ = Date et heure départ de la course Sprint (ou nul)
            """
        )
        
        df = pd.read_csv(r'../data/races.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Results
    st.markdown(
        """
        ### Results

        La table « Results » fournit les résultats des Grands Prix organisés dans l’histoire de la Formule 1.

        __Note__ : Les données sur le tour le plus rapide des pilotes sont disponibles à partir de la saison _2004_.
        """
    )
    if st.checkbox('Voir la table « Results »'):
        st.markdown(
            """
            - __resultId__ = ID (clé primaire)
            - __raceId__ = ID du Grand Prix
            - __driverId__ = ID du pilote
            - __constructorId__ = ID écurie
            - __number__ = Numéro du pilote
            - __grid__ = Position sur la grille de départ
            - __position__ = Position à la ligne d'arrivée (ou nul)
            - __positionText__ = Idem 'position' au format texte
            - __positionOrder__ = Numéro pour tri
            - __points__ = Points gagnés par le pilote à l'issue de la course
            - __laps__ = Nombre de tours complétés
            - __time__ = Temps total cumulé à l'issue de la course pour le pilote arrivé 1er / écart par rapport au 1er
            - __milliseconds__ = Temps total cumulé converti en millisecondes
            - __fastestLap__ = Numéro du tour le plus rapide réalisé par le pilote
            - __rank__ = Classement du tour le plus rapide par rapport aux autres pilotes
            - __fastestLapTime__ = Chrono du tour le plus rapide réalisé par le pilote
            - __fastestLapSpeed__ = Vitesse (km/h) du tour le plus rapide
            - __statusId__ = ID statut classement final (arrivé/abandon…)
            """
        )
        
        df = pd.read_csv(r'../data/results.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Seasons
    st.markdown(
        """
        ### Seasons

        La table « Seasons » fournit les informations sur les Championnats organisés dans l’histoire de la Formule 1.
        """
    )
    if st.checkbox('Voir la table « Seasons »'):
        st.markdown(
            """
            - __year__ = Année du championnat
            - __url__ = Page Wikipédia
            """
        )
        
        df = pd.read_csv(r'../data/seasons.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Sprint results
    st.markdown(
        """
        ### Sprint results

        La table « Sprint results » fournit les résultats des séances Sprint des pilotes.

        C’est une mini-course introduite en 2021 sur 3 Grands Prix en test et reconduite pour 2022.

        Le Sprint se déroule après la séance de qualifications dont le classement final constitue la grille de départ du Sprint. Le classement à l'arrivée du Sprint définit la grille de départ de la course.
        """
    )
    if st.checkbox('Voir la table « Sprint results »'):
        st.markdown(
            """
            - __sprintResultId__ = ID (clé primaire)
            - __raceId__ = ID du Grand Prix
            - __driverId__ = ID du pilote
            - __constructorId__ = ID écurie
            - __number__ = Numéro du pilote
            - __grid__ = Position grille de départ
            - __position__ = Position arrivé (ou nul)
            - __positionText__ = Idem au format texte
            - __positionOrder__ = Numéro pour tri
            - __points__ = Points gagnés par le pilote à l'issue du Sprint
            - __laps__ = Nombre de tours complétés
            - __time__ = Temps total cumulé à l'issue du Sprint pour le pilote arrivé 1er / écart par rapport au 1er
            - __milliseconds__ = Temps total cumulé converti en millisecondes
            - __fastestLap__ = Numéro du tour le plus rapide réalisé par le pilote
            - __fastestLapTime__ = Chrono du tour le plus rapide réalisé par le pilote
            - __statusId__ = ID statut classement final (arrivé/abandon…)
            """
        )
        
        df = pd.read_csv(r'../data/sprint_results.csv')

        st.dataframe(df.head())

    st.markdown('---')

    # Table Status
    st.markdown(
        """
        ### Status

        La table « Status » fournit les informations sur les différents statuts à l'arrivée de la course/sprint.
        """
    )
    if st.checkbox('Voir la table « Status »'):
        st.markdown(
            """
            - __statusId__ = ID (clé primaire)
            - __status__ = Label du statut (arrivé, abandon, disqualifié…)
            """
        )
        
        df = pd.read_csv(r'../data/status.csv')

        st.dataframe(df.head())