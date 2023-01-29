import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image


title = "Classement sur une saison"
sidebar_name = "Classement 2021"


def run():

    st.markdown('<style>.stProgress div[role="progressbar"] > div > div {height: 1.6rem; background-color: unset;} .stProgress div[role="progressbar"] > div > div > div {background-color: #e10600;} section[tabindex="0"] > div > div:nth-child(1) > div > div:nth-child(20) > div {gap:0.5rem;} section[tabindex="0"] > div > div:nth-child(1) > div > div:nth-child(20) div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) div[data-testid="stMarkdownContainer"] {font-family: "Zen Dots"; border-left: 3px solid #38383f;} section[tabindex="0"] > div > div:nth-child(1) > div > div:nth-child(20) div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) div[data-testid="stMarkdownContainer"] > p {padding: 2px 0 0 7px; font-size: 0.9rem; text-align: center;} section[tabindex="0"] > div > div:nth-child(1) > div > div:nth-child(20) div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) div[data-testid="stMarkdownContainer"] {font-family: "Zen Dots"; background-color: #d0d0d2; border-bottom-right-radius: 5px;} section[tabindex="0"] > div > div:nth-child(1) > div > div:nth-child(20) div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) div[data-testid="stMarkdownContainer"] > p {font-size: 0.9rem; color: #15151e; font-weight: 700; text-align: center;} section[tabindex="0"] > div > div:nth-child(1) > div > div:nth-child(20) > div > div:nth-child(1) {margin-bottom: 1rem;} </style>', unsafe_allow_html=True)

    st.title(title)

  
    points = pd.read_csv(r'..\data\points_classement_2021.csv',sep=',',index_col = 0)
    races = pd.read_csv(r'..\data\races.csv',sep=',')
    results = pd.read_csv(r'..\data\results.csv',sep=',')
    drivers = pd.read_csv(r'..\data\drivers.csv',sep=',')
    rfc_proba = pd.read_csv(r'..\data\rfc_proba_complet.csv',sep=',',index_col=0)
    svc_proba = pd.read_csv(r'..\data\svc_proba_complet.csv',sep=',',index_col=0)
    logreg_proba = pd.read_csv(r'..\data\logreg_proba_complet.csv',sep=',',index_col=0)
    tree_proba = pd.read_csv(r'..\data\decisiontree_proba_complet.csv',sep=',',index_col=0)
    compar_svc = pd.read_csv(r'..\data\compar_svc_class_complet.csv',sep=',',index_col=0)
    compar_rfc = pd.read_csv(r'..\data\compar_rfc_class_complet.csv',sep=',',index_col=0)
    compar_logreg = pd.read_csv(r'..\data\compar_logreg_class_complet.csv',sep=',',index_col=0)
    compar_tree = pd.read_csv(r'..\data\compar_decisiontreee_class_complet.csv',sep=',',index_col=0)
    
    
    st.markdown(
        """
        L'idée de cette page est d'essayer de prédire le classement complet de chaque course du Championnat sur la saison 2021.

        ## Modélisation et récupération des probabilités

        Comme précédemment, on part donc de notre dataframe. On demande maintenant à nos modèles de nous donner la probabilité que chaque pilote finisse à chacune des 20 positions. Voici par exemple ce que l'on obtient :
        - Chaque colonne « _proba_n_ » représentant la probabilité d'être en n-ième position
        - Chaque ligne représentant un pilote _(driverId)_ lors d'un GP donné _(raceId)_ comme on peut le voir ici
        """)
    st.dataframe(rfc_proba.head())
    
    st.markdown(
        """
        ---
        
        ## Méthodologie

        La méthodologie est alors la suivante :
        - Regrouper par course/raceId.
        - Choisir le premier GP de l'année.
        - Choisir le pilote qui a la plus forte probabilité d'être premier sur ce Grand Prix.
        - Lui attribuer la première place et les 25 points associés.
        - Pour le deuxième, nous avions plusieurs options :
            - Prendre celui avec la __proba_2__ la plus élevée.
            - Additionner __proba_1__ et __proba_2__ et choisir le pilote avec la somme la plus élevée, **c'est ce que nous avons fait.**
        - Attribuer alors les 18 points au pilote.
        - Ainsi de suite jusqu'à la 20e position.
        - Recommencer pour le Grand Prix n°2 de l'année.
        - Faire la somme totale des points après chaque course.
        
        Ce qui nous donne un  nouveau dataframe :
        """)
     
    st.dataframe(points.head())
    st.markdown(
        """
        Chaque pilote (_driverId_) se voit attribuer un classement et le nombre de points associés pour une course donnée : _classement_1052_ et _points_1052_ correspondent à la course dont l'id est 1052, c'est-à-dire le Grand Prix de Bahrein.
        
        On calcule aussi les points totaux après chaque course : _points_totaux_XXX_

        Sur la base des points dans la colonne _points_totaux_1073_ qui correspond à la dernière course. Il nous reste alors à établir le classement final prédit et le comparer au classement réel.
        """)
    
    st.markdown(
        """
        ---

        ## Résultats

        Ce qui nous donne en fonction de nos modèles :
        """)
 
    ##### AFFICHAGE des ecart de points et classement
    
    plt.rcParams['figure.facecolor'] = '#15151e'
    plt.rcParams['axes.facecolor'] = '#0e1117'
    models = ['SVC','Random Forest Classifier','Régression Logistique','Arbre de décision']
    ecart_point  = [round(compar_svc['diff_points'].abs().mean(), 1),
                    round(compar_rfc['diff_points'].abs().mean(),1),
                    round(compar_logreg['diff_points'].abs().mean(),1),
                    round(compar_tree['diff_points'].abs().mean(),1)]
    
    ecart_place  = [round(compar_svc['diff_classement'].abs().mean(),1),
                    round(compar_rfc['diff_classement'].abs().mean(),1),
                    round(compar_logreg['diff_classement'].abs().mean(),1),
                    round(compar_tree['diff_classement'].abs().mean(),1)]
    
    color_models  = [ '#4c78a8', '#4c78a8', '#4c78a8','#e10600']
    
    fig_1 = plt.figure(figsize=(15, 3))
    plt.bar(x=models, height=ecart_point, width=0.6, color=color_models)
    plt.title('Ecart de points moyen en fonction du modèle')
    plt.ylabel('Ecart de points')
    plt.xticks(rotation=45, ha='right')
    for i in range(len(ecart_point)):
         plt.annotate(str(ecart_point[i]), xy=(models[i], ecart_point[i]), ha='center', va='bottom', color='#fff')
    st.pyplot(fig_1)
    
    fig_2 = plt.figure(figsize=(15, 3))
    plt.bar(x=models, height=ecart_place, width=0.6, color=color_models)
    plt.title('Ecart de classement en fonction du modèle')
    plt.ylabel('Ecart de classement')
    plt.xticks(rotation=45, ha='right')
    for i in range(len(ecart_point)):
         plt.annotate(str(ecart_place[i]), xy=(models[i], ecart_place[i]), ha='center', va='bottom', color='#fff')
    st.pyplot(fig_2) 
    
    st.write("On voit très rapidement que le modèle d'arbre de décision est très performant. ")
    
    
    def df_background_color_class_ok(s):
        if s['classement_final'] == s['classement_final_reel']:
            return ['background-color: #386641']*len(s)
        elif abs(s['classement_final'] - s['classement_final_reel']) == 1:
            return ['background-color: #002B5C']*len(s)
        else:
            return ['background-color: #0e1117']*len(s)
    
        
    with st.expander("Pour avoir le détail des dataframes :"):
        st.markdown(
            """
            - _Les deux premières colonnes _points_totaux_1073_ et _classement_final_ présentent les résultats du modèle._
            - _Les colonnes _points_reels_ et _classement_final_reel_ sont les vrais résultats._
            - _Les deux colonnes de droite présentent les différences entre les résultats réels et ceux prédits._

            _La ligne apparait en <span style="color:#32cd32">vert</span> si le classement du pilote prédit est le même que le classement réel._

            _La ligne apparait en <span style="color:#00bfff">bleu</span> si la différence entre la prédiction et la réalite est d'une seule place._ 

            _Vous pouvez cliquer sur le nom d'une colonne pour l'ordonner (classé selon classement final réel par défaut)._
            """,unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            linear = st.checkbox('Modèle Linear SVC', key='0_linear')
        with col2:
            rfc = st.checkbox('Modèle Random Forest Classifier',key='0_rfc')
        with col3:
            log_reg = st.checkbox('Modèle Logistic Regression',key='0_log_reg')
        with col4:
            tree = st.checkbox('Modèle Arbre de décision',key='0_tree')
    
        if linear:  
            ## import dataframe SVC
            st.subheader("Linear SVC")
            st.write("Ecart de points moyen : "+str(round(compar_svc['diff_points'].abs().mean(),1)))
            st.write("Ecart de classement moyen : "+str(round(compar_svc['diff_classement'].abs().mean(),1)))
            st.dataframe(compar_svc.sort_values(by='classement_final_reel').style.apply(df_background_color_class_ok, axis=1))
           
            
        if rfc:
            ## import dataframe RFC
            st.subheader("Random Forest Classifier")
            st.write("Ecart de points moyen : "+str(round(compar_rfc['diff_points'].abs().mean(),1)))
            st.write("Ecart de classement moyen : "+str(round(compar_rfc['diff_classement'].abs().mean(),1)))
            st.dataframe(compar_rfc.sort_values(by='classement_final_reel').style.apply(df_background_color_class_ok, axis=1))
    
        
        if log_reg:
             ## import dataframe LogReg
             st.subheader("Régression logistique")    
             st.write("Ecart de points moyen : "+str(round(compar_logreg['diff_points'].abs().mean(),1)))
             st.write("Ecart de classement moyen : "+str(round(compar_logreg['diff_classement'].abs().mean(),1)))
             st.dataframe(compar_logreg.sort_values(by='classement_final_reel').style.apply(df_background_color_class_ok, axis=1))
    
        if tree:  
            ## import dataframe SVC
            st.subheader("Decision Tree")
            st.write("Ecart de points moyen : "+str(round(compar_tree['diff_points'].abs().mean(),1)))
            st.write("Ecart de classement moyen : "+str(round(compar_tree['diff_classement'].abs().mean(),1)))
            st.dataframe(compar_tree.sort_values(by='classement_final_reel').style.apply(df_background_color_class_ok, axis=1))


    st.write("             ")
    st.markdown(
        """
        ---

        ## Représentation graphique
        """)
    st.write("Une image valant mille dataframes, voici les représentations graphiques associées comparant les prédictions des modèles et les points réellement engrangés par les pilotes :")
    

    with st.expander("Pour avoir les représentations graphiques :"):
        ### Choix des modèles à afficher
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            linear = st.checkbox('Modèle Linear SVC')
        with col2:
            rfc = st.checkbox('Modèle Random Forest Classifier')
        with col3:
            log_reg = st.checkbox('Modèle Logistic Regression')
        with col4:
            tree = st.checkbox('Modèle Arbre de décision')
            
        ### Test
        st.write("        ")
        st.write("        ")
        if linear:
     
            ## import image SVC
            st.subheader("Linear SVC - Ordonné selon le classement réel")
            image_svc = Image.open(r'..\data\classement_2021_svc.png')
            st.image(image_svc)
            
            st.subheader("Linear SVC - Ordonné selon le classement prédit")
            image_svc_2 = Image.open(r'..\data\classement_2021_svc_2.png')
            st.image(image_svc_2)
            
            st.write("             ")
            st.write("On remarque rapidement que ce modèle ne semble pas optimal, voire pas bon du tout. Ce que confirme l'écart moyen de 88 points. "
                     "Il semble que celui-ci privilégie les anciens champions du monde ou en d'autres termes les pilotes avec le plus de victoires "
                     "en Championnat, en effet, il place Raikkonen, Hamilton, Vettel et Alonso en tête. ")
            st.write(      "Sans surprise on observe des écarts de points entre les prédictions et la réalité allant jusqu'à 433... ce qui est énorme. "
                     "Pour aller plus loin, il faudrait utiliser un outil comme [Shap](https://shap.readthedocs.io/en/latest/) pour "
                     "tenter de comprendre le comportement du modèle.")
        if rfc:
            
            ## import image RFC
            st.subheader("Random Forest Classifier - Ordonné selon le classement réel")
            image_rfc = Image.open(r'..\data\classement_2021_rfc.png')
            st.image(image_rfc)
            
            st.subheader("Random Forest Classifier - Ordonné selon le classement prédit")
            image_rfc_2 = Image.open(r'..\data\classement_2021_rfc_2.png')
            st.image(image_rfc_2)
            st.write("             ")
            st.write("Pour les deux premières places, les résultats sont très bons ! "
                     "On observe un écart de seulement 1.5 point pour Verstappen et 16.5 pour Hamilton soient respectivement 0.3% et 4.2% d'écart ! "
                     "L'écart moyen est seulement de 29.5 points contre 88 précédemment. ")
            st.write("Le plus grand écart de points est de 124 et cela concerne Pérez, qui a, d'après ce modèle, clairement sous-performé. "
                     "Mais cela peut s'expliquer : En regardant de plus près Perez n’a pas terminé 6 courses sur les 22 "
                     "(ce qui est un chiffre très élevé pour une écurie comme RedBull). Sur les 16 courses restantes, "
                     "il a marqué en moyenne 11.9 points, ce qui ramené à une saison de 22 courses aurait fait une saison "
                     "à 262 points. Même si on est encore loin des 314 points prédits, cela l’aurait quand même placé en 3ème position, comme prédit par le modèle.")
    
        if log_reg:
            
            ## import image logreg
            st.subheader("Régression Logistique - Ordonné selon le classement réel")
            image_logreg = Image.open(r'..\data\classement_2021_log_reg.png')
            st.image(image_logreg)
            
            st.subheader("Régression Logistique - Ordonné selon le classement prédit")
            image_logreg_2 = Image.open(r'..\data\classement_2021_log_reg_2.png')
            st.image(image_logreg_2)
            st.write("             ")
            st.write("Ce modèle semble se comporter un peu comme le modèle SVC en plaçant Raikkonnen, Alonso et Vettel dans les 6 premiers. "
                     "Cependant il semble mieux prendre en compte le présent en plaçant Hamilton, Verstappen et Perez aussi dans le top 6.")
            st.write("Comme pour SVC on observe des écarts de points entre les prédictions et la réalité allant jusqu'à 253. "
                        "Ici aussi l'utilisation de SHAP permettrait de comprendre quelles features sont les plus mises en avant par le modèle. "
                        "Par pondération on pourrait alors tenter d'optimiser le modèle. ")
        if tree:
            
            ## import image tree
            st.subheader("Decision Tree - Ordonné selon le classement réel")
            image_tree = Image.open(r'..\data\classement_2021_decision_tree.png')
            st.image(image_tree)
            
            st.subheader("Decision Tree - Ordonné selon le classement prédit")
            image_tree_2 = Image.open(r'..\data\classement_2021_decision_tree_2.png')
            st.image(image_tree_2)
            st.write("C'est de très loin le meilleur modèle ! Encore meilleur que le Random Forest, l'écart moyen est de 10.5 points! "
                     "Les plus grands écarts de points sont pour Verstappen et Hamilton (49 et 43 points), le 3ème plus grand n'est que de 28 points. ! "
                     "Point intéressant, sur les 21 pilotes, seuls 4 ont été surévalués (Russel, Gasly, Latifi, Vettel). Il semble que le modèle "
                     "ait une tendance à clairement sous-évaluer les performances des pilotes. ")
            st.write("Au niveau du classement le Top 7 est correctement prédit, dans l'ordre qui plus est. Les positions 8 et 9, "
                     "ainsi que 10 et 11 sont simplement inversées. ")
        


    
    
    ####################################### FIN SAISON
    ###################################### Début Proba 1 Pilote
    ## à partir de 2011.
    
    
    st.markdown(
        """
        ---

        ## Probabilités de victoire pour un pilote

        Puisque nous avons calculé les probabilités à chaque course, nous pouvons aussi les présenter de manière plus visuelle.

        Nous allons représenter la probabilité qu'a chaque pilote de finir à une position donnée. Il faut donc choisir le modèle, l'année, puis le Grand Prix, puis le pilote.


        """)

    
    col1, col2, col3, col4 = st.columns(4)

    with col1: 
        model = st.selectbox(label='Choix du modèle', options = ['Linear SVC','Random Forest Classifier','Logistic Regression','Decision Tree'], key='0_model')
        if model == 'Linear SVC':
            st.write("[page Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)")
            df_proba = svc_proba
        elif model == 'Logistic Regression':
            st.write("[page Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)")
            df_proba = logreg_proba 
        elif model == 'Decision Tree':
            st.write("[page Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)")
            df_proba = tree_proba
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
    
    #réduction du tableau
    liste_surname = drivers[drivers['driverId'].isin(liste_driverId)]['surname'].sort_values(ascending=True)
    
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