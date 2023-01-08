import streamlit as st
import pandas as pd
# import numpy as np
# import time
# from joblib import dump, load
# from sklearn import svm
# from sklearn.svm import SVC

# from sklearn.svm import LinearSVC
# from sklearn.calibration import CalibratedClassifierCV
# from imblearn.over_sampling import RandomOverSampler, SMOTE
# from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from PIL import Image
# import raceplotly
# from raceplotly.plots import barplot

title = "Classement sur une saison"
sidebar_name = "Classement 2021"


def run():

    st.markdown('<style>.stProgress div[role="progressbar"] div.st-bs {height: 1.6rem; background-color: unset;} .stProgress div[role="progressbar"] div.st-bn {background-color: #e10600;} section.main.css-1v3fvcr.egzxvld3 > div > div:nth-child(1) > div > div:nth-child(55) > div.css-1xh5rm1 {gap:0.5rem;} section.main.css-1v3fvcr.egzxvld3 > div > div:nth-child(1) > div > div:nth-child(55) > div.css-1xh5rm1 div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) div[data-testid="stMarkdownContainer"] {font-family: "Zen Dots"; border-left: 3px solid #38383f;} section.main.css-1v3fvcr.egzxvld3 > div > div:nth-child(1) > div > div:nth-child(55) > div.css-1xh5rm1 div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) div[data-testid="stMarkdownContainer"] > p {padding: 2px 0 0 7px; font-size: 0.9rem; text-align: center;} section.main.css-1v3fvcr.egzxvld3 > div > div:nth-child(1) > div > div:nth-child(55) > div.css-1xh5rm1 div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) div[data-testid="stMarkdownContainer"] {font-family: "Zen Dots"; background-color: #d0d0d2; border-bottom-right-radius: 5px;} section.main.css-1v3fvcr.egzxvld3 > div > div:nth-child(1) > div > div:nth-child(55) > div.css-1xh5rm1 div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) div[data-testid="stMarkdownContainer"] > p {font-size: 0.9rem; color: #15151e; font-weight: 700; text-align: center;} section.main.css-1v3fvcr.egzxvld3 > div > div:nth-child(1) > div > div:nth-child(55) > div.css-1xh5rm1 > div:nth-child(1) {margin-bottom: 1rem;} </style>', unsafe_allow_html=True)
    
    st.title(title)

    st.markdown(
        """
        L'idée de cette page est d'essayer de prédire le classement complet de chaque course et du Championnat sur la saison 2021.
         
        """
    )

    #df_results = pd.read_csv(r'..\data\df_results_class_complet.csv',sep=',')
   # data_scaled = pd.read_csv(r'..\data\data_scaled_class_complet.csv',sep=',')
    #df_drivers = pd.read_csv(r'..\data\drivers.csv',sep=',')
    # on importe depuis un csv le vrai classement
    #resultats_2021=pd.read_csv(r'..\data\Classement_2021.csv', encoding = 'utf-8',sep=';')
    points = pd.read_csv(r'..\data\points_classement_2021.csv',sep=',',index_col = 0)
    y_proba = pd.read_csv(r'..\data\proba_class_complet.csv',sep=',',index_col = 0)
    races = pd.read_csv(r'..\data\races.csv',sep=',')
    results = pd.read_csv(r'..\data\results.csv',sep=',')
    drivers = pd.read_csv(r'..\data\drivers.csv',sep=',')
    rfc_proba = pd.read_csv(r'..\data\rfc_proba_complet.csv',sep=',',index_col=0)
    svc_proba = pd.read_csv(r'..\data\svc_proba_complet.csv',sep=',',index_col=0)
    
    st.markdown(
        """
        ## Application du modèle
        """)
    st.write("Comme précédemment on part donc de notre dataframe. On demande maintenant à nos modèles de nous donner la probabilité "
             "que chaque pilote finisse à chacune des 20 positions. Voici par exemple ce que l'on obtient : ")
    st.write("Chaque colonne 'proba_j' représentant la probabilité d'être en 'j_ème' position")
    st.write("Chaque ligne représentant un pilote _(driverId)_ lors d'un GP donné _(raceId)_ comme on peut le voir ici")
    st.dataframe(rfc_proba.head())
    
    st.markdown(
        """
        ## Méthodologie
        """)
    st.write("La méthodologie est alors la suivante :")
    st.write("    - regrouper par course/raceId. ")
    st.write("    - choisir le premier GP de l'année. ")
    st.write("    - choisir le pilote qui a la plus forte probabilité d'être premier sur ce GP. ")
    st.write("    - lui attribuer la première place et les 25 points associés. ")
    st.write("    - pour le deuxième, nous avions plusieurs options : ")
    st.write("       - - prendre celui avec la proba_2 la plus élevée. ")
    st.write("       - - additionner proba_1 et proba_2 et choisir le pilote avec la somme la plus élevée, **c'est ce que nous avons fait.** ")
    st.write("    - attribuer alors les 18 points au pilote. ")  
    st.write("    - ainsi de suite jusqu'à la 20eme position. ")
    st.write("    - recommencer pour le GP no 2 de l'année. ")
    st.write("    - faire la somme totale des points après chaque course. ")
    st.write("   ")
    st.write("Ce qui nous donne un  nouveau dataframe : ")
     
    st.dataframe(points.head())
    st.write("Chaque pilote (driverId) se voit attribuer un classement et le nombre de points associés pour une course donnée"
             " _classement_1052_ et _points_1052_ correspondent à la course dont l'id est 1052, c'est à dire le Gp de Bahrein. "
             "On calcule aussi les points totaux après chaque course : _points_totaux_XXX_ ")
    
    st.write("Sur la base des points dans la colonne _points_totaux_1073_ qui correspond à la dernière course. "
             "Il nous reste alors à établir le classement final prédit et le comparer au classement réel. ")
    
    st.markdown(
        """
        ## Résultats
        """)
        
    st.write("Ce qui nous donne pour les modèles LinearSVC et Random Forest : ")
    
    st.write("_Info : vous pouvez cliquer sur le nom d'une colonne pour l'ordonner_ ")
    col1, col2 = st.columns(2)

    with col1:  
        ## import dataframe SVC
        st.subheader("Linear SVC")
        compar_svc = pd.read_csv(r'..\data\compar_svc_class_complet.csv',sep=',',index_col=0)
        st.dataframe(compar_svc)
        st.write("L'écart de points moyen est de ", compar_svc['diff_points'].abs().mean())
    with col2:
        ## import dataframe RFC
        st.subheader("Random Forest Classifier")
        compar_rfc = pd.read_csv(r'..\data\compar_rfc_class_complet.csv',sep=',',index_col=0)
        st.dataframe(compar_rfc)
        compar_rfc['diff_points'].abs().mean()
        st.write("L'écart de points moyen est de ", compar_rfc['diff_points'].abs().mean())
    
    st.write("Les deux premières colonnes _points_totaux_1073_ et _classement_final_ présentent les résultats du modèle. " 
             "Les colonnes _points_reels_ et _classement_final_reel_ sont les vrais résultats. "
             "Les deux colonnes de droite présentent les différences entre les résultats réels et ceux prédits.")
    st.write("             ")
    st.write("             ")
    st.write("Une image valant mille mots, voici les représentations graphiques associées : ")
    

 
    ## import image SVC
    st.header("Linear SVC - Ordonné selon le classement réel")
    image_svc = Image.open(r'..\data\classement_2021_svc.png')
    st.image(image_svc)
    
    st.header("Linear SVC - Ordonné selon le classement prédit")
    image_svc_2 = Image.open(r'..\data\classement_2021_svc_2.png')
    st.image(image_svc_2)
    
    st.write("             ")
    st.write("On remarque rapidement que ce modèle ne semble pas optimal, voire pas bon du tout. Ce que confirme l'écart moyen de 88 points."
             "Il semble que celui-ci privilégie les anciens champions du monde ou en d'autres termes les pilotes avec le plus de victoires "
             "en Championnat, en effet, il place Raikkonen, Hamilton, Vettel et Alonso en tête. ")
    st.write(      "Sans surprise on observe des écarts de points entre les prédictions et la réalité allant jusqu'à 433...ce qui est énorme. "
             "Pour aller plus loin, il faudrait utiliser un outil comme [Shap](https://shap.readthedocs.io/en/latest/) pour "
             "tenter de comprendre le comportement du modèle.")

    ## import image RFC
    st.header("Random Forest Classifier - Ordonné selon le classement réel")
    image_rfc = Image.open(r'..\data\classement_2021_rfc.png')
    st.image(image_rfc)
    
    st.header("Random Forest Classifier - Ordonné selon le classement prédit")
    image_rfc_2 = Image.open(r'..\data\classement_2021_rfc_2.png')
    st.image(image_rfc_2)
    st.write("             ")
    st.write("Pour les deux premmières places, les résultats sont très bons ! "
             "On observe un écart de seulement 1.5 points pour Verstappen et 16.5 pour Hamilton soient respectivement 0.3% et 4.2% d'écart ! "
             "L'écart moyen est seulement de 29.5 points contre 88 précédemment. ")
    st.write("Le plus grand écart de points est de 124 et cela concerne Pérez, qui a, d'après ce modèle, clairement sous-performé. "
             "Mais cela peut s'expliquer : En regardant de plus près Perez n’a pas terminé 6 courses sur les 22 "
             "(ce qui est un chiffre très élevé pour une écurie comme RedBull). Sur les 16 restantes, "
             "il a marqué en moyenne 11.9 points, ce qui ramené à une saison de 22 courses aurait fait une saison "
             "à 262 points. Même si on est encore loin des 314 points prédits, cela l’aurait quand même placé en 3ème position, comme prédit par le modèle.")

    
    ####################################### FIN SAISON
    ###################################### Début Proba 1 Pilote
    ## à partir de 2011.
    
    
    st.markdown(
        """
        # Probabilités de victoire pour un pilote 
        """)
        
    st.write("Puisque nous avons calculé les probabilités à chaque course, nous pouvons aussi les présenter de manière plus visuelle. ")
    st.write("Nous allons représenter la probabilité qu'a chaque pilote de finir à une position donnée. "
             "Il faut donc choisir le modèle, l'année, puis le Grand Prix, puis le pilote.")
    st.write("")

    
    col1, col2, col3, col4 = st.columns(4)

    with col1: 
        model = st.selectbox(label='Choix du modèle', options = ['Linear SVC','Random Forest Classifier'], key='0_model')
        if model == 'Linear SVC':
            st.write("[page Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)")
            df_proba = svc_proba
        else :
            df_proba = rfc_proba
            st.write("[page Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")
        
    with col2:    
        ## choix de l'année
        annee = st.selectbox(label='Année', options=[2021,2020,2019,2018,2017,2016,2015,2014,2013,2012],key='0_annee') 
        ## récupération du circuit en fonction de l'année
        liste_course = races[races['year'] == annee]['name']
        st.write("[page Wiki du Championnat](https://fr.wikipedia.org/wiki/Championnat_du_monde_de_Formule_1_"+str(annee)+")")
    with col3: 
        #choix de la course
        course = st.selectbox(label='Course', options=liste_course,key='0_course')
        
        # récupération de raceId
        race_id = races[(races['year']== annee) & (races['name']== course)][['raceId']].iloc[0,0]
        st.write("[page Wiki du GP]("+races[races['raceId']==race_id][['url']].iloc[0,0]+")")
        
    #récupération des pilotes dans la course avec ce raceID
    liste_driverId = results[results['raceId'] == race_id]['driverId'].tolist()
    #st.write(liste_driverId)
    
    #réduction du tableau
    liste_surname = drivers[drivers['driverId'].isin(liste_driverId)]['surname'].sort_values(ascending=True)
    #st.write(type(liste_surname))
    
    with col4: 
        driver_1 = st.selectbox(label='Choix du pilote 1', options = liste_surname, key='0_driver1')
        
        ## beaucoup d'homonymes ou de familles dans la f1...
        if driver_1 == 'Verstappen':
            driver_id = 830
        elif driver_1 == 'Schumacher':
            if race_id > 880:
                driver_id = 854
            else:
                driver_id = 30
        elif driver_1 =='Hartley':
            driver_id = 843
        elif driver_1 =='Fittipaldi':
            driver_id = 850
        elif driver_1 =='Magnussen':
            driver_id = 825
        elif driver_1 =='Palmer':
            driver_id = 835
        elif driver_1 =='Bianchi':
            driver_id = 824
        elif driver_1 =='Rosberg':
            driver_id = 3
        elif driver_1 =='Nakajima':
            driver_id = 6
        else:
            driver_id =  drivers[drivers['surname'] == driver_1][["driverId"]].iloc[0,0]
        
        st.write("[page Wiki du pilote]("+drivers[drivers['driverId']==driver_id][['url']].iloc[0,0]+")")
    
    with st.container():
        
        st.subheader("Position / Estimation")
        
        for i in range(1,21):
            temp = df_proba[(df_proba['raceId'] == race_id) & (df_proba['driverId'] == driver_id)]['proba_'+str(i)].iloc[0]
            
            col1, col2, col3 = st.columns([0.05,0.12,1])
            with col1:
                st.write(str(i))
            with col2: 
                st.markdown(f'{round(temp*100,2)} %')
            with col3:
                my_bar = st.progress(0)
                my_bar.progress(temp)
        
    # position = []
    # pourcent = []
    # barre = []
    # for i in range(1,21):
    #     temp = df_proba[(df_proba['raceId'] == race_id) & (df_proba['driverId'] == driver_id)]['proba_'+str(i)].iloc[0]
    #     position.append("Position "+str(i))
    #     pourcent.append(round(temp*100,2))
    #     #barre.append(my_bar.progress(temp))
    #     df=pd.DataFrame()
    #     df['position'] = position
    #     df['pourcent'] = pourcent
    #     df['barre'] = 0
    # df.iloc[0,2]=my_bar.progress(50)
    # st.write(df.iloc[0,2])
    # st.dataframe(df.head(20))
    