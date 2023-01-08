import streamlit as st

import pandas as pd
import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import plotly.express as px


import fastf1 as ff1
ff1.Cache.enable_cache(cache_dir=r'tabs\cache')    #on met tout en cache pr que ca charge plus vite plus tard
from fastf1 import plotting
ff1.plotting.setup_mpl() 
from fastf1 import utils

import datetime


title = "Dataviz"
sidebar_name = "DataViz'"


def run():
    


    st.title(title)
    st.write('Cette page va présenter divers graphiques exploitant nos jeux de données :')
    st.write("    - le premier jeu de données sera constitué de la librairie [fastf1](https://theoehrly.github.io/Fast-F1/) "
             "dont on utilisera certaines fonctions. "
             " A noter que cette librairie ne proposent des données qu'à partir de 2018. C'est pourquoi nous "
             "ne l'avons pas utilisé dans nos modèles. Néanmoins il nous semble intéressant de voir les possibilités qu'elle offre. ")
    st.write("**_A noter que la quantité de données à télécharger pour chaque graphique est relativement conséquente. Si vous "
             "êtes le premier à demander l'affichage d'un duo 'année/course' alors l'affichage devrait prendre "
             "une vingtaine de secondes environ. Un peu moins sinon._**")
    st.write('    - le second jeu de données sera constitué de certains dataframes présentés précédemment')
    
    ######### Début Partie  - Map des courses dans l'histoire




    st.markdown(
            """
            ## Où se sont déroulées toutes les courses de F1 de l'histoire ?

            """
        )
    circuits = pd.read_csv(r'..\data\circuits.csv',sep=',')
    races = pd.read_csv(r'..\data\races.csv',sep=',')
    
    st.write("On souhaite présenter sur une carte l'emplacement de tous les Grands Prix de l'histoire")
    st.write('Pour cela, on a au départ deux dataframes issus de notre jeu de données qui ressemblent à ca : ')
    
    
    st.write('Le dataframe Circuits')
    st.dataframe(circuits.head())

    st.write('Le dataframe Races')
    st.dataframe(races.head())

    st.write('On ne garde que les infos qui nous intéressent et on arrive à ces deux dataframes : ')   
    
    
    circuits = circuits.drop(['circuitRef','url'],axis=1)
    races = races.drop(['raceId','round','time','url','fp1_date','fp1_time','fp2_date','fp2_time',
                        'fp3_date','fp3_time','quali_date','quali_time','sprint_date','sprint_time'],axis=1)
    
    st.write('Le dataframe Circuits')
    st.dataframe(circuits.head())
    st.write('Le dataframe Races')
    st.dataframe(races.head())
    
    
    st.write('Finalement, on les regroupe en un seul dataframe : ') 
    
    
    df_races_final=circuits.merge(races, left_on='circuitId',right_on='circuitId',
                        suffixes=('_circuits','_races'))
    
    #renomme la colone name_races
    df_races_final.rename(columns = {'name_races':'Nom de la course'}, inplace = True)    
    
    st.dataframe(df_races_final.sample(5))   

    st.write("On est alors capable d'associer une année avec une latitude, une longitude et un nom de Grand Prix.") 
    

    year_min=1950   
    year_max=2022
    map_projections = ['equirectangular', 'mercator', 'orthographic', 'natural earth', 'kavrayskiy7',
                       'miller', 'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant',
                       'conic equal area', 'conic conformal', 'conic equidistant', 'gnomonic', 'stereographic',
                       'mollweide', 'hammer', 'transverse mercator', 'albers usa', 'winkel tripel', 'aitoff',
                       'sinusoidal']
    
    #création du slider
    saison_min,saison_max=st.slider("Choisissez les années voulues ", min_value=year_min, max_value=year_max,value=(year_min,year_max))
    #choix de la projection
    projection = st.selectbox(label='Choisissez la projection (pour les cartophiles) ', options=map_projections, help="Quelle carte ?",key='projection')
    #affichage              
    affichage_circuit = px.scatter_geo(data_frame=df_races_final[(df_races_final['year']<=saison_max) & (df_races_final['year']>=saison_min)] ,
                                       lat='lat', lon='lng', title='Emplacement des circuits dans le monde',
                               hover_name='name_circuits',
                               hover_data ={'country','location'},
                               color='Nom de la course',
                               projection =projection)
    
    affichage_circuit.update_layout(height=800)
    st.plotly_chart(affichage_circuit, use_container_width=True)
    
    st.write("On peut remarquer que dès la première saison en 1950, une course se déroulait en Amérique du Nord, la fameuse Indy 500. "
             "Suivi de près par l'Amérique du Sud en 1953. La première course en Afrique se déroula, elle, au Maroc en 1958. "
             "Contrairement aux critiques récentes, on constate clairement que la F1 a toujours eu vocation à s'exporter. "
             "La première course en Afrique se déroula au Maroc en 1958. "
             "C'est le Japon qui hébergea la première course en Asie de l'histoire, en 1976. "
             "9 ans plus tard les Formules 1 débarquèrent en Océanie, à Adélaide plus précisément.") 
    ######### Fin Partie  - Map des courses dans l'histoire
    
######### Début Partie  - Course le jour de votre birthday

    st.markdown(
             """
             ## Quelle course a eu lieu le jour de votre anniversaire ?
             
    
             """
     )   
    st.write("Après avoir vu **où** se déroulait les courses, il est maintenant intéréssant de savoir **quand** elles se sont déroulées.")
    st.write('Une petite transformation à notre tableau pour ajouter les colonnes "mois" et "jour" : ')
    
    df_races_final['date'] = pd.to_datetime(df_races_final['date']).dt.date
    df_races_final['day'] = df_races_final['date'].map(lambda dt: dt.strftime('%d')).astype(int)
    df_races_final['month'] = df_races_final['date'].map(lambda dt: dt.strftime('%m')).astype(int)
    

    
    
    st.dataframe(df_races_final.sample(5)) 
    
    date = st.date_input("Choisissez maintenant votre jour de naissance.",min_value=datetime.date(1950, 1, 1),max_value=datetime.date(2029,1 , 1))
    
    #liste_day_birth = range(1,32)
    #liste_month_birth = range(1,13)
    #day_birth = st.selectbox(label='Votre jour de naissance', options=liste_day_birth)
    #month_birth = st.selectbox(label='Votre mois de naissance', options=liste_month_birth)
    
    day_birth = date.day
    month_birth = date.month
    
    
    if st.button(label="Alors ?",args=None, help ="ça arrive...", disabled=False):
        st.write("Voici la liste des courses s'étant déroulé le jour de votre naissance : ")
        st.dataframe(df_races_final[(df_races_final['day']==day_birth) & (df_races_final['month']==month_birth)][['circuitId', 'Nom de la course','name_circuits','location','country','date']])
        
        #affichage               
        affichage_circuit_2 = px.scatter_geo(data_frame=df_races_final[(df_races_final['day']==day_birth) & (df_races_final['month']==month_birth)] ,
                                           lat='lat', lon='lng', title='Emplacement des courses au jour sélectionné',
                                   hover_name='name_circuits',
                                   hover_data ={'country','location'},
                                   color='Nom de la course',
                                   projection =projection)
        
        affichage_circuit_2.update_layout(height=700)
        st.plotly_chart(affichage_circuit_2, use_container_width=True)

######### Fin Partie  - Course le jour de votre birthday    
    
    





    ########## Début  Partie      --- Comparaison chrono sur la course 
    st.markdown(
        """
        ## Comparaison tour par tour du chrono de deux pilotes durant une course.
        """
    )  
    st.write("Dans cette première utilisation de fastf1 nous allons présenter les chronos de deux pilotes tout au long d'une course. ")
    st.write("Il faut donc choisir l'année, puis le Grand Prix, puis les pilotes.")
    races = pd.read_csv(r'..\data\races.csv',sep=',')
    results = pd.read_csv(r'..\data\results.csv',sep=',')
    drivers = pd.read_csv(r'..\data\drivers.csv',sep=',')
    
   
    col1, col2, col3, col4 = st.columns(4)

    with col1:    
        ## choix de l'année
        annee = st.selectbox(label='Année', options=[2022,2021,2020,2019,2018],key='0_annee') 
        
        ## récupération du circuit en fonction de l'année
        liste_course = races[races['year'] == annee]['name']
    with col2: 
        #choix de la course
        course = st.selectbox(label='Course', options=liste_course,key='0_course')
        
        # récupération de raceId
        race_id = races[(races['year']== annee) & (races['name']== course)][['raceId']].iloc[0,0]
        #st.write(race_id)

    #récupération des pilotes dans la course avec ce raceID
    liste_driverId = results[results['raceId'] == race_id]['driverId'].tolist()
    #st.write(liste_driverId)
    
    #réduction du tableau
    #temp = 
    liste_surname = drivers[drivers['driverId'].isin(liste_driverId)]['surname'].sort_values(ascending=True)
    #st.write(type(liste_surname))
    
    with col3: 
        driver_1 = st.selectbox(label='Choix du pilote 1', options = liste_surname, key='0_driver1')
    with col4: 
        driver_2 = st.selectbox(label='Choix du pilote 2', options = liste_surname, key='0_driver2')
    #st.write(driver_1)
    
    
    if driver_1 == 'Verstappen':
        trigramme_driver1 = 'VER'
    elif driver_1 == 'Schumacher':
        trigramme_driver1 = 'MSC'
    else:
        trigramme_driver1 =  drivers[drivers['surname'] == driver_1]['code'].iloc[0]
        
    if driver_2 == 'Verstappen':
        trigramme_driver2 = 'VER'
    elif driver_2 == 'Schumacher':
        trigramme_driver2 = 'MSC'
    else:
        trigramme_driver2 =  drivers[drivers['surname'] == driver_2]['code'].iloc[0]

    #st.write(trigramme_driver1)
    #st.write(trigramme_driver2)
    
    if st.button(label="Visualisation", key = 'visu_lap', args = None, help ="L'affichage peut prendre un peu de temps", disabled=False):     
                
        race = ff1.get_session(annee, course, 'R')
        race.load()
    
        driver1 = race.laps.pick_driver(trigramme_driver1)
        driver2 = race.laps.pick_driver(trigramme_driver2)
        color_driver_1 = "#"+race.get_driver(trigramme_driver1).TeamColor
        color_driver_2 = "#"+race.get_driver(trigramme_driver2).TeamColor 
        
        fig, ax = plt.subplots()
        ax.plot(driver1['LapNumber'], driver1['LapTime'], color=color_driver_1, label=f'{driver_1}')
        ax.plot(driver2['LapNumber'], driver2['LapTime'], color=color_driver_2, label=f'{driver_2}')
        ax.set_title(f"{race.event.OfficialEventName}\n"
                        f" Comparaison tour par tour\n"
                        f"{driver_1} Vs {driver_2}")
        ax.set_xlabel("Tour")
        ax.set_ylabel("Chrono")
        ax.legend()
        st.pyplot(fig)
        st.write("_Les tours où le pilote est passé au stand changer ses pneus apparaissent souvent **très** clairement._ ")
        st.write("_Les tours où les données sont manquantes correspondent très souvent aux tours où le pilote a été impliqué dans un accident._ ")
        
    
  ########## Fin   Partie     --- Comparaison chrono sur la course   
 


  ########## Début    Partie      --- Comparaison télémétrie
  
    st.markdown(
        """
        ## Comparaison de la télémétrie de deux pilotes en course.
        """
    )
    st.write('Pour cette deuxième utilisation de fastf1, nous allons comparer la **télémetrie** de deux pilotes sur leur meilleur tour en course.')
    st.write("Ici aussi, il faut donc choisir l'année, puis le Grand Prix, puis les pilotes.")
    st.write("")
    st.write("Il en ressort alors deux graphiques, un présentant la vitesse en fonction du temps "
             "et l'autre la vitesse en fonction de la distance.")
    
    
    
    
    races = pd.read_csv(r'..\data\races.csv',sep=',')
    results = pd.read_csv(r'..\data\results.csv',sep=',')
    drivers = pd.read_csv(r'..\data\drivers.csv',sep=',')
    
   
    col1, col2, col3, col4 = st.columns(4)  
    with col1:
        ## choix de l'année
        annee = st.selectbox(label='Année', options=[2022,2021,2020,2019,2018],key='1_annee') 
        
    ## récupération du circuit en fonction de l'année
    liste_course = races[races['year'] == annee]['name']
    with col2:
        #choix de la course
        course = st.selectbox(label='Course', options=liste_course,key='1_course')
        
    # récupération de raceId
    race_id = races[(races['year']== annee) & (races['name']== course)][['raceId']].iloc[0,0]
    #st.write(race_id)

    #récupération des pilotes dans la course avec ce raceID
    liste_driverId = results[results['raceId'] == race_id]['driverId'].tolist()
    #st.write(liste_driverId)
    
    #réduction du tableau
    #temp = 
    liste_surname = drivers[drivers['driverId'].isin(liste_driverId)]['surname'].sort_values(ascending=True)
    #st.write(type(liste_surname))
    
    with col3:
        driver_1 = st.selectbox(label='Choix du pilote 1', options = liste_surname, key='1_driver1')
    with col4:
        driver_2 = st.selectbox(label='Choix du pilote 2', options = liste_surname, key='1_driver2')
    #st.write(driver_1)
    
    
    if driver_1 == 'Verstappen':
        trigramme_driver1 = 'VER'
    elif driver_1 == 'Schumacher':
        trigramme_driver1 = 'MSC'
    else:
        trigramme_driver1 =  drivers[drivers['surname'] == driver_1]['code'].iloc[0]
        
    if driver_2 == 'Verstappen':
        trigramme_driver2 = 'VER'
    elif driver_2 == 'Schumacher':
        trigramme_driver2 = 'MSC'
    else:
        trigramme_driver2 =  drivers[drivers['surname'] == driver_2]['code'].iloc[0]
    
    
    session_type = 'R'  # R= race ; P1=Practice 1 ; ...
    

    if st.button(label="Visualisation",args=None, help ="L'affichage peut prendre un peu de temps", disabled=False, key='1_visu'):
        
        
        session = ff1.get_session(annee, course, session_type)
        session.load()
    
        # on charge les données pour le driver 1
    
        #test = session.laps.pick_driver(driver_1)
    
        #renvoie les infos du tour le plus rapide : objet fastf1.core.lap
        fast_driver_1 = session.laps.pick_driver(trigramme_driver1).pick_fastest() 
    
        #renvoie les car data : objet fastf1.core.Telemetry
        driver_1_car_data = fast_driver_1.get_car_data().add_distance()  
    
        t_driver_1 = driver_1_car_data['Time']                                #pandas.core.series.Series
        d_driver_1 = driver_1_car_data['Distance']                            #pandas.core.series.Series
        vit_Car_driver_1 = driver_1_car_data['Speed']                         #pandas.core.series.Series
        color_driver_1 = "#"+session.get_driver(trigramme_driver1).TeamColor           #string
        driver1_telemetry = fast_driver_1.get_telemetry()
    
        # on charge les données pour le driver 2
    
        fast_driver_2 = session.laps.pick_driver(trigramme_driver2).pick_fastest()
        driver_2_car_data = fast_driver_2.get_car_data().add_distance()
        t_driver_2 = driver_2_car_data['Time']
        d_driver_2 = driver_2_car_data['Distance']
        vit_Car_driver_2 = driver_2_car_data['Speed']
        color_driver_2 = "#"+session.get_driver(trigramme_driver2).TeamColor
        driver2_telemetry = fast_driver_2.get_telemetry()
    
       
        ## on dessine
    
        fig_tel, ax_tel = plt.subplots(2,figsize=(15,25))
    
    ## dessin de la vitesse en fonction du temps
    
        ax_tel[0].plot(t_driver_1, vit_Car_driver_1, label=f'{driver_1}', color=color_driver_1)
        ax_tel[0].plot(t_driver_2, vit_Car_driver_2, label=f'{driver_2}', color=color_driver_2)
        ax_tel[0].set_xlabel('Temps')
        ax_tel[0].set_ylabel('Vitesse [Km/h]')
        ax_tel[0].set_title(f"{session.event.OfficialEventName}\n"
                        f" Comparaison seconde par seconde\n"
                        f"{driver_1} Vs {driver_2}")
        
        
        ax_tel[0].legend()
        ax_tel[0].set_facecolor('#262730')


        ## dessin de la distance en fonction du temps
        ax_tel[1].plot(d_driver_1, vit_Car_driver_1, label=f'{driver_1}', color=color_driver_1)
        ax_tel[1].plot(d_driver_2, vit_Car_driver_2, label=f'{driver_2}', color=color_driver_2)
        
        ## création du nouvel axe y via la fonction twinx()
        delta_time, ref_tel, compare_tel = utils.delta_time(fast_driver_1, fast_driver_2)
        twin = ax_tel[1].twinx()
        twin.plot(ref_tel['Distance'], delta_time, '--', color='white')
        twin.axhline(y=0,color='white',linestyle='--')
 
        
        twin.set_ylabel("<-- " f"{driver_2} plus rapide | {driver_1}"" plus rapide -->")
        
        
        ax_tel[1].set_xlabel('Distance')
        ax_tel[1].set_ylabel('Vitesse [Km/h]')
        ax_tel[1].set_title(f"{session.event.OfficialEventName}\n"
                        f" Comparaison mètre par mètre\n"
                        f"{driver_2} Vs {driver_1}")
        
        ax_tel[1].legend()
        ax_tel[1].set_facecolor('#262730')
        
        
        st.write(f"Temps de {driver_1}\n"
                             f"{str(fast_driver_1.LapTime)[10:-3]}", f"et Temps de {driver_2}\n"
                                                   f"{str(fast_driver_2.LapTime)[10:-3]}")
        fig_tel.patch.set_facecolor('#262730')
        
        st.pyplot(fig_tel)
        
        # st.write(f"temps de {driver_2}\n"
        #                       f"{str(fast_driver_2.LapTime)[10:-3]}",color=color_driver_2)

    



########## Fin Partie - Télémétrie deux pilotes
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
######### Début Partie  - Dégradé de vitesse lors du Tour le plus rapide

    st.markdown(
        """
        
        
        
        
        
        ## Visualisation de la vitesse sur le circuit lors de son tour le plus rapide

         """
    )
    st.write("Dans cette dernière utilisation de fastf1, nous allons présenter l'évolution de la vitesse sur le circuit"
             "lors du tour le plus rapide du pilote")
    st.write("Il faut donc choisir l'année, puis le Grand Prix, puis le pilote.")
    races = pd.read_csv(r'..\data\races.csv',sep=',')
    results = pd.read_csv(r'..\data\results.csv',sep=',')
    drivers = pd.read_csv(r'..\data\drivers.csv',sep=',')
    
   
    col1, col2, col3= st.columns(3)  
    with col1:
        ## choix de l'année
        annee = st.selectbox(label='Année', options=[2022,2021,2020,2019,2018],key='2_annee') 
    
    ## récupération du circuit en fonction de l'année
    liste_course = races[races['year'] == annee]['name']
    
    with col2:
        #choix de la course
        course = st.selectbox(label='Course', options=liste_course,key='2_course')
    
    # récupération de raceId
    race_id = races[(races['year']== annee) & (races['name']== course)][['raceId']].iloc[0,0]
    #st.write(race_id)

    #récupération des pilotes dans la course avec ce raceID
    liste_driverId = results[results['raceId'] == race_id]['driverId'].tolist()
    #st.write(liste_driverId)
    
    #réduction du tableau
    #temp = 
    liste_surname = drivers[drivers['driverId'].isin(liste_driverId)]['surname'].sort_values(ascending=True)
    #st.write(type(liste_surname))
    
    with col3:   
        driver_1 = st.selectbox(label='Choix du pilote 1', options = liste_surname, key='2_driver1')

    #st.write(driver_1)
    
    
    if driver_1 == 'Verstappen':
        trigramme_driver1 = 'VER'
    elif driver_1 == 'Schumacher':
        trigramme_driver1 = 'MSC'
    else:
        trigramme_driver1 =  drivers[drivers['surname'] == driver_1]['code'].iloc[0]
        
        
    
    session_type = 'R'  # R= race ; P1=Practice 1 ; ...
    #création de l'array des coordonnées x et y et des couleurs
    
    if st.button(label="Visualisation",key='2_visu', args=None, help ="L'affichage peut prendre un peu de temps", disabled=False):
        
        session = ff1.get_session(annee, course, session_type)
        session.load()
        
        fast_driver_1 = session.laps.pick_driver(trigramme_driver1).pick_fastest()
        driver1_telemetry = fast_driver_1.get_telemetry()
        x1 = np.array(driver1_telemetry['X'])
        y1 = np.array(driver1_telemetry['Y'])
        color1=np.array(driver1_telemetry['Speed'])
        points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
        color_driver_1 = "#"+session.get_driver(trigramme_driver1).TeamColor
        segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
        colormap = cm.plasma
        
        #on créé la figure et le titre
        fig_vit, ax_vit = plt.subplots(figsize=(6,6))
        ax_vit.set_title(f"{session.event.OfficialEventName}\n"
                        f" Vitesse lors du meilleur tour\n"
                        f"{driver_1}")
        ax_vit.axis('off')
        
        #on affiche les données
        #le circuit
        ax_vit.plot(x1, y1, color='white', linestyle='-', linewidth=7, zorder=0)
       
        # on lisse la couleur 
        # on "linéarise" l'ensemble des points
        norm1 = plt.Normalize(color1.min(), color1.max())
        lc1 = LineCollection(segments1, cmap=colormap, norm=norm1, linestyle='-', linewidth=5)
        
        # définition de la valeur qui va définir la couleur, ici color1 donc en réalité la vitesse 
        lc1.set_array(color1)
        
        # on relie tous les segments entre eux
        line1 = ax_vit.add_collection(lc1)
        
        # Création de la colorbar
        cbaxes = fig_vit.add_axes([0.25, 0.05, 0.5, 0.05])
        normlegend = mpl.colors.Normalize(vmin=color1.min(), vmax=color1.max())
        legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal", )


        # Affichage de la figure
        #ff1.plotting.setup_mpl() 
        
        fig_vit.patch.set_facecolor('black')
        st.pyplot(fig_vit)




######### Fin Partie  - Dégradé de vitesse lors du Tour le plus rapide
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

######### Début Partie 5 ????