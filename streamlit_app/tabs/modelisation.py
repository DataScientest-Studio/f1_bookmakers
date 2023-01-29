import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

title = "Modélisation - Vainqueur"
sidebar_name = "Modélisation - Vainqueur"

def run():

    # fonction pour les boutons avec session_state
    def stateful_button(*args, key=None, **kwargs):
        if key is None:
            raise ValueError("Must pass key")

        if key not in st.session_state:
            st.session_state[key] = False

        if st.button(*args, **kwargs):
            st.session_state[key] = not st.session_state[key]

        return st.session_state[key]

    st.title(title)

    # -----------------------
    # préparation des données
    # -----------------------

    # chargement données
    df = pd.read_csv(r"../data/df_results_meteo_circuit_classement.csv", sep=';', index_col=0)
    # chargement données pilotes
    drivers_data = pd.read_csv(r"../data/drivers.csv")

    # suppression des lignes avec nan
    df = df.dropna()

    df_columns = ['driverId', 'constructorId', 'grid', 'fastestLapSpeed_classes', 'positionOrder', 'driverStandingPosition', 'driverWins', 'constructorStandingPosition', 'constructorWins']
    df[df_columns] = df[df_columns].astype('int')

    # modif valeurs positionOrder à 1 pour prédire le gagnant (position = 1) sinon 0 pour les autres valeurs
    df['positionOrder'] = df['positionOrder'].apply(lambda x: 1 if x==1 else 0)


    # jeux données train/test
    df_train = df[df['year']<=2020]
    df_test = df[df['year']==2021]

    # séparation données features / target
    X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
    y_train = df_train['positionOrder']
    X_test = df_test.drop(['year', 'round', 'positionOrder'], axis=1)
    y_test = df_test['positionOrder']

    # normalisation des données
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # rééchantillonnage (Régression Log + Arbre de décision)
    ros = RandomOverSampler()
    X_ro, y_ro = ros.fit_resample(X_train_scaled, y_train)
    

    st.markdown(
        """
        Dans cette partie, nous allons aborder la méthodologie de modélisation pour prédire le vainqueur des Grands Prix.

        ## Méthodologie

        En partant de notre dataframe, la variable cible est la colonne « positionOrder ».

        Afin de déterminer le vainqueur :
        - La position 1 aura la valeur 1
        - Les autres positions auront la valeur 0
        
        Voici par exemple ce que l'on obtient :
        """)
    st.dataframe(df.reset_index().head(20))
    st.markdown(
        """
        Cela engendre un déséquilibre des données avec la valeur 0 comme classe majoritaire. On utilise donc une méthode de rééchantillonnage pour équilibrer les données et avoir un ratio classes minoritaire / majoritaire satisfaisant (nous avons utilisée la méthode **RandomOverSampler**).
        
        Pour les prédictions, on récupère des modèles la probabilité que chaque pilote finisse à la première place. Le vainqueur sera celui avec la plus forte probabilité de gagner.

        <ul><li><b>Etape 1</b> : calcul des probabilités
        
        Avec la fonction <b>predict_proba()</b>, nous récupérons les probabilités des classes du modèle :
        
        La colonne **proba_0** indique la probabilité que le pilote ne finisse pas la course 1er. Inversement la colonne **proba_1** indique la probabilité que le pilote soit gagnant.</li></ul>
        """, unsafe_allow_html=True)
    st.image(r'./assets/modelisation_vainqueur_proba_etape1.jpg')
    st.markdown(
        """
        <ul><li><b>Etape 2</b> : définition du vainqueur pour chaque course
        
        Pour déterminer le vainqueur d’un Grand prix :

        <ul><li>Les probabilités sont fusionnées avec le jeu de données test.</li>
        <li>Une colonne « prediction », initialisée à 0, est rajoutée.</li></ul>
        """, unsafe_allow_html=True)
    st.image(r'./assets/modelisation_vainqueur_proba_etape2.jpg')
    st.markdown(
        """
        <ul><li><b>Etape 3</b> : prédiction
        
        Une boucle est appliquée pour tous les grands Grand Prix (ou « raceId ») :
        
        <ul><li>On regroupe par raceId</li>
        <li>La valeur maximale de la probabilité pour la classe 1 (colonne <b>proba_1</b>) est identifiée</li>
        <li>Sur cette ligne, on définit la valeur 1 dans la colonne « prediction »</li></ul>
        """, unsafe_allow_html=True)
    st.image(r'./assets/modelisation_vainqueur_proba_etape3.jpg')
    st.markdown(
        """
        Cette méthodologie nous permet d’avoir un vainqueur pour chaque Grand Prix dans les prédictions et ce même si les probabilités de la classe 1 ne dépassent pas les 50% sur un même Grand Prix.

        ---
        """, unsafe_allow_html=True)
    st.markdown('<style> section[tabindex="0"] > div > div:nth-child(1) > div > div > div > div > div > img {margin-left: 3rem;} </style>', unsafe_allow_html=True)

    
    # ----------------------------
    # Algorithmes
    # ----------------------------
    st.markdown(
        """
        ## Prédictions

        Nous avons ciblé le championnat 2021 comme échantillon de test pour les prédictions.

        ### Itération 1
        
        Les données sont départagées de la manière suivante :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse.
        - Jeu de test : les données de l’année 2021

        """)

    model_selector = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision', 'SVC', 'KNN'), key="iter1",
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)
    
    if model_selector == 'Régression logistique':
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter1, param_col2_iter1, param_col3_iter1, param_col4_iter1 = st.columns(4)

        with param_col1_iter1:
            C_param_selector = st.selectbox(label='C', options=(0.001, 0.01, 0.1, 1, 10), index=1, key='log-iter1')

        if stateful_button(label='Résultats itération 1 - Régression logistique', key='button-log-iter1'):
        # if st.button('Résultats', key='log-iter1'):  

            st.write('---')

            # instanciation modèle
            log_reg = LogisticRegression(C=C_param_selector)
            log_reg.fit(X_ro, y_ro)

            # probabilité avec predict_proba
            y_pred_log_ro2 = log_reg.predict_proba(X_test_scaled)
            df_y_pred_log_ro2 = pd.DataFrame(y_pred_log_ro2, columns=['proba_0', 'proba_1'])


            # création dataframe des résultats
            df_test_proba = pd.concat([df_test.reset_index(), df_y_pred_log_ro2], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_proba['prediction'] = 0


            # liste des courses par raceId
            raceId_list = df_test_proba['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list:
                df_temp = df_test_proba[df_test_proba['raceId']==i]
                max_proba_1 = df_temp['proba_1'].max()
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index
                df_test_proba.loc[index_max_proba_1, 'prediction'] = 1
            

            # rapport classification et matrice de confusion
            confusion_matrix = pd.crosstab(df_test_proba['positionOrder'], df_test_proba['prediction'])
            confusion_matrix.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix.index = ['Real val. 0', 'Real val. 1']

            classif_report_df = pd.DataFrame(classification_report(y_test, df_test_proba['prediction'], output_dict=True)).T[:2]
            classif_report_df['support'] = classif_report_df['support'].astype('int')
            classif_report_df.index = ['Class 0', 'Class 1']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_df)


            # dataframe avec les pilotes vainqueurs réels
            winner_real_log = df_test_proba[df_test_proba["positionOrder"]==1][['round','driverId']]
            # dataframe avec les pilotes vainqueurs dans les prédicitons
            winner_predicted_log = df_test_proba[df_test_proba["prediction"]==1][['round','driverId']]

            # fusion des données pilotes dans les 2 dataframes
            winner_real_log = winner_real_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Winner'})\
                                                .drop(['driverId'], axis=1)
            winner_predicted_log = winner_predicted_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .drop(['driverId'], axis=1)
            # fusion des 2 dataframes
            winners_results_log = winner_real_log.merge(right=winner_predicted_log, on='round').sort_values(by=['round']).reset_index(drop=True)
            winners_results_log['match'] = winners_results_log.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


            with col2_iter1:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_log, height=735)
    

    elif model_selector == 'Forêt aléatoire':
        # ----------------------
        # Modèle Forêt aléatoire
        # ----------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter1, param_col2_iter1, param_col3_iter1, param_col4_iter1 = st.columns(4)

        with param_col1_iter1:
            n_estimators_param_selector = st.selectbox(label='n_estimators', options=(10, 50, 100, 250), index=2, key='rf_param1-iter1')
        with param_col2_iter1:
            min_samples_leaf_param_selector = st.selectbox(label='min_samples_leaf', options=(1, 3, 5), index=0, key='rf_param2-iter1')
        with param_col3_iter1:
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param3-iter1')

        if stateful_button(label='Résultats itération 1 - Forêt aléatoire', key='button-rf-iter1'):
        # if st.button('Résultats', key='rf-iter1'):

            st.write('---')
        
            # instanciation modèle
            rf = RandomForestClassifier(n_jobs=-1, max_features = max_features_param_selector, min_samples_leaf = min_samples_leaf_param_selector,
                                            n_estimators = n_estimators_param_selector, random_state=1430)
            rf.fit(X_train_scaled, y_train)

            # probabilité avec predict_proba
            y_pred_rf_proba = rf.predict_proba(X_test_scaled)
            df_y_pred_rf_proba = pd.DataFrame(y_pred_rf_proba, columns=['proba_0', 'proba_1'])


            # création dataframe des résultats
            df_test_prob_rf = pd.concat([df_test.reset_index(), df_y_pred_rf_proba], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_rf['prediction'] = 0


            # liste des courses par raceId
            raceId_listRF = df_test_prob_rf['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_listRF:
                df_temp = df_test_prob_rf[df_test_prob_rf['raceId']==i]
                max_proba_1 = df_temp['proba_1'].max()
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index
                df_test_prob_rf.loc[index_max_proba_1, 'prediction'] = 1
            

            # rapport classification et matrice de confusion
            confusion_matrix_rf = pd.crosstab(df_test_prob_rf['positionOrder'], df_test_prob_rf['prediction'])
            confusion_matrix_rf.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_rf.index = ['Real val. 0', 'Real val. 1']

            classif_report_rf_df = pd.DataFrame(classification_report(y_test, df_test_prob_rf['prediction'], output_dict=True)).T[:2]
            classif_report_rf_df['support'] = classif_report_rf_df['support'].astype('int')
            classif_report_rf_df.index = ['Class 0', 'Class 1']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_rf)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_rf_df)


            # dataframe avec les pilotes vainqueurs réels
            winner_real_rf = df_test_prob_rf[df_test_prob_rf["positionOrder"]==1][['round','driverId']]
            # dataframe avec les pilotes vainqueurs dans les prédicitons
            winner_predicted_rf = df_test_prob_rf[df_test_prob_rf["prediction"]==1][['round','driverId']]

            # fusion des données pilotes dans les 2 dataframes
            winner_real_rf = winner_real_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Winner'})\
                                                .drop(['driverId'], axis=1)
            winner_predicted_rf = winner_predicted_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .drop(['driverId'], axis=1)
            # fusion des 2 dataframes
            winners_results_rf = winner_real_rf.merge(right=winner_predicted_rf, on='round').sort_values(by=['round']).reset_index(drop=True)
            winners_results_rf['match'] = winners_results_rf.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


            with col2_iter1:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_rf, height=735)
    

    elif model_selector == 'Arbre de décision':
        # ------------------------
        # Modèle Arbre de décision
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter1, param_col2_iter1, param_col3_iter1, param_col4_iter1 = st.columns(4)

        with param_col1_iter1:
            criterion_param_selector = st.selectbox(label='criterion', options=('entropy', 'gini'), index=0, key='dt_param1-iter1')
        with param_col2_iter1:
            max_depth_param_selector = st.selectbox(label='max_depth', options=(1, 2, 3, 4, 5, 6, 7), index=4, key='dt_param2-iter1')

        if stateful_button(label='Résultats itération 1 - Arbre de décision', key='button-dt-iter1'):
        # if st.button('Résultats', key='dt-iter1'):

            st.write('---')

            # instanciation modèle
            dt_clf = DecisionTreeClassifier(criterion=criterion_param_selector, max_depth=max_depth_param_selector, random_state=143)
            dt_clf.fit(X_ro, y_ro)

            # probabilité avec predict_proba
            y_pred_dt_proba = dt_clf.predict_proba(X_test_scaled)
            df_y_pred_dt_proba = pd.DataFrame(y_pred_dt_proba, columns=['proba_0', 'proba_1'])


            # création dataframe des résultats
            df_test_prob_dt = pd.concat([df_test.reset_index(), df_y_pred_dt_proba], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_dt['prediction'] = 0


            # liste des courses par raceId
            raceId_list_dt = df_test_prob_dt['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list_dt:
                df_temp = df_test_prob_dt[df_test_prob_dt['raceId']==i]
                max_proba_1 = df_temp['proba_1'].max()
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index
                df_test_prob_dt.loc[index_max_proba_1, 'prediction'] = 1
            

            # rapport classification et matrice de confusion
            confusion_matrix_dt = pd.crosstab(df_test_prob_dt['positionOrder'], df_test_prob_dt['prediction'])
            confusion_matrix_dt.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_dt.index = ['Real val. 0', 'Real val. 1']

            classif_report_dt_df = pd.DataFrame(classification_report(y_test, df_test_prob_dt['prediction'], output_dict=True)).T[:2]
            classif_report_dt_df['support'] = classif_report_dt_df['support'].astype('int')
            classif_report_dt_df.index = ['Class 0', 'Class 1']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_dt)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_dt_df)


            # dataframe avec les pilotes vainqueurs réels
            winner_real_dt = df_test_prob_dt[df_test_prob_dt["positionOrder"]==1][['round','driverId']]
            # dataframe avec les pilotes vainqueurs dans les prédicitons
            winner_predicted_dt = df_test_prob_dt[df_test_prob_dt["prediction"]==1][['round','driverId']]

            # fusion des données pilotes dans les 2 dataframes
            winner_real_dt = winner_real_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Winner'})\
                                                .drop(['driverId'], axis=1)
            winner_predicted_dt = winner_predicted_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .drop(['driverId'], axis=1)
            # fusion des 2 dataframes
            winners_results_dt = winner_real_dt.merge(right=winner_predicted_dt, on='round').sort_values(by=['round']).reset_index(drop=True)
            winners_results_dt['match'] = winners_results_dt.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


            with col2_iter1:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_dt, height=735)
    

    elif model_selector == 'SVC':
        # ------------------------
        # Modèle SVC
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter1, param_col2_iter1, param_col3_iter1, param_col4_iter1 = st.columns(4)

        with param_col1_iter1:
            C_param_selector = st.selectbox(label='C', options=(0.05, 0.1, 1, 10), index=1, key='svc_param1-iter1')
        with param_col2_iter1:
            kernel_param_selector = st.selectbox(label='kernel', options=('linear', 'poly', 'rbf'), index=0, key='svc_param2-iter1')

        if stateful_button(label='Résultats itération 1 - SVC', key='button-svc-iter1'):
        # if st.button('Résultats', key='svc-iter1'):

            st.write('---')

            # instanciation modèle
            svc_ro = SVC(C=C_param_selector, kernel=kernel_param_selector, probability=True)
            svc_ro.fit(X_ro,y_ro)

            # probabilité avec predict_proba
            y_pred_svc_ro_proba = svc_ro.predict_proba(X_test_scaled)
            df_y_pred_svc_ro_proba = pd.DataFrame(y_pred_svc_ro_proba, columns=['proba_0', 'proba_1'])


            # création dataframe des résultats
            df_test_prob_svc_ro = pd.concat([df_test.reset_index(), df_y_pred_svc_ro_proba], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_svc_ro['prediction'] = 0


            # liste des courses par raceId
            raceId_list_svc_ro = df_test_prob_svc_ro['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list_svc_ro:
                df_temp = df_test_prob_svc_ro[df_test_prob_svc_ro['raceId']==i]
                max_proba_1 = df_temp['proba_1'].max()
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index
                df_test_prob_svc_ro.loc[index_max_proba_1, 'prediction'] = 1
            

            # rapport classification et matrice de confusion
            confusion_matrix_svc_ro = pd.crosstab(df_test_prob_svc_ro['positionOrder'], df_test_prob_svc_ro['prediction'])
            confusion_matrix_svc_ro.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_svc_ro.index = ['Real val. 0', 'Real val. 1']

            classif_report_svc_ro_df = pd.DataFrame(classification_report(y_test, df_test_prob_svc_ro['prediction'], output_dict=True)).T[:2]
            classif_report_svc_ro_df['support'] = classif_report_svc_ro_df['support'].astype('int')
            classif_report_svc_ro_df.index = ['Class 0', 'Class 1']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_svc_ro)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_svc_ro_df)


            # dataframe avec les pilotes vainqueurs réels
            winner_real_svc_ro = df_test_prob_svc_ro[df_test_prob_svc_ro["positionOrder"]==1][['round','driverId']]
            # dataframe avec les pilotes vainqueurs dans les prédicitons
            winner_predicted_svc_ro = df_test_prob_svc_ro[df_test_prob_svc_ro["prediction"]==1][['round','driverId']]

            # fusion des données pilotes dans les 2 dataframes
            winner_real_svc_ro = winner_real_svc_ro.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Winner'})\
                                                .drop(['driverId'], axis=1)
            winner_predicted_svc_ro = winner_predicted_svc_ro.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .drop(['driverId'], axis=1)
            # fusion des 2 dataframes
            winners_results_svc_ro = winner_real_svc_ro.merge(right=winner_predicted_svc_ro, on='round').sort_values(by=['round']).reset_index(drop=True)
            winners_results_svc_ro['match'] = winners_results_svc_ro.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


            with col2_iter1:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_svc_ro, height=735)


    elif model_selector == 'KNN':
        # ------------------------
        # Modèle KNN
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter1, param_col2_iter1, param_col3_iter1, param_col4_iter1 = st.columns(4)

        with param_col1_iter1:
            n_neighbors_param_selector = st.selectbox(label='n_neighbors', options=(2, 5, 7, 10), index=3, key='knn_param1-iter1')
        with param_col2_iter1:
            metric_param_selector = st.selectbox(label='metric', options=('minkowski', 'manhattan', 'chebyshev'), index=1, key='knn_param2-iter1')

        if stateful_button(label='Résultats itération 1 - KNN', key='button-knn-iter1'):
        # if st.button('Résultats', key='knn-iter1'):

            st.write('---')

            # instanciation modèle
            knn_ro = KNeighborsClassifier(n_neighbors=n_neighbors_param_selector, metric=metric_param_selector)
            knn_ro.fit(X_ro,y_ro)

            # probabilité avec predict_proba
            y_pred_knn_ro_proba = knn_ro.predict_proba(X_test_scaled)
            df_y_pred_knn_ro_proba = pd.DataFrame(y_pred_knn_ro_proba, columns=['proba_0', 'proba_1'])


            # création dataframe des résultats
            df_test_prob_knn_ro = pd.concat([df_test.reset_index(), df_y_pred_knn_ro_proba], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_knn_ro['prediction'] = 0


            # liste des courses par raceId
            raceId_list_knn_ro = df_test_prob_knn_ro['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list_knn_ro:
                df_temp = df_test_prob_knn_ro[df_test_prob_knn_ro['raceId']==i]
                max_proba_1 = df_temp['proba_1'].max()
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index
                df_test_prob_knn_ro.loc[index_max_proba_1, 'prediction'] = 1
            

            # rapport classification et matrice de confusion
            confusion_matrix_knn_ro = pd.crosstab(df_test_prob_knn_ro['positionOrder'], df_test_prob_knn_ro['prediction'])
            confusion_matrix_knn_ro.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_knn_ro.index = ['Real val. 0', 'Real val. 1']

            classif_report_knn_ro_df = pd.DataFrame(classification_report(y_test, df_test_prob_knn_ro['prediction'], output_dict=True)).T[:2]
            classif_report_knn_ro_df['support'] = classif_report_knn_ro_df['support'].astype('int')
            classif_report_knn_ro_df.index = ['Class 0', 'Class 1']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_knn_ro)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_knn_ro_df)


            # dataframe avec les pilotes vainqueurs réels
            winner_real_knn_ro = df_test_prob_knn_ro[df_test_prob_knn_ro["positionOrder"]==1][['round','driverId']]
            # dataframe avec les pilotes vainqueurs dans les prédicitons
            winner_predicted_knn_ro = df_test_prob_knn_ro[df_test_prob_knn_ro["prediction"]==1][['round','driverId']]

            # fusion des données pilotes dans les 2 dataframes
            winner_real_knn_ro = winner_real_knn_ro.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Winner'})\
                                                .drop(['driverId'], axis=1)
            winner_predicted_knn_ro = winner_predicted_knn_ro.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .drop(['driverId'], axis=1)
            # fusion des 2 dataframes
            winners_results_knn_ro = winner_real_knn_ro.merge(right=winner_predicted_knn_ro, on='round').sort_values(by=['round']).reset_index(drop=True)
            winners_results_knn_ro['match'] = winners_results_knn_ro.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


            with col2_iter1:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_knn_ro, height=735)



    st.write('---')

    st.markdown(
        """
        ### Itération 2

        Nous souhaitions voir s’il était possible d’ajouter les données des courses passées dans le jeu d’entrainement à chaque course et observer les résultats obtenus.

        Pour la <u>première course</u> du championnat, nous avons la répartition des données comme suit :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse.
        - Jeu de test : les données de la 1ere course du championnat 2021.

        Pour la <u>deuxième course</u>, la répartition serait la suivante :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse + les données de la 1ere course du championnat 2021.
        - Jeu de test : les données de la 2e course du championnat 2021.

        Pour la <u>troisième course</u>, nous aurions :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse + les données de la 1ere et 2e courses du championnat 2021.
        - Jeu de test : les données de la 3e course du championnat 2021.

        Et ainsi de suite.

        Les modèles sont réajustés à chaque mise à jour des jeux d’entrainements et les résultats cumulés au fur et à mesure des courses.

        """, unsafe_allow_html=True)
    
    model_selector_2 = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision', 'KNN'), key='iter2',
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)

    if model_selector_2 == 'Régression logistique':
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter2, param_col2_iter2, param_col3_iter2, param_col4_iter2 = st.columns(4)

        with param_col1_iter2:
            C_param_selector = st.selectbox(label='C', options=(0.001, 0.01, 0.1, 1, 10), index=1, key='log-iter2')

        if stateful_button(label='Résultats itération 2 - Régression logistique', key='button-log-iter2'):
        # if st.button('Résultats', key='log-iter2'):  

            st.write('---')

            # initialisation données features / target
            X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
            y_train = df_train['positionOrder']

            # instanciation fonction de normalisation des données
            scaler = StandardScaler().fit(X_train)

            # instanciation modèle
            log_reg = LogisticRegression(C=C_param_selector)

            # initialisation dataframe compilation des vainqueurs réels et prédits
            df_winner = pd.DataFrame(columns=['round', 'Winner', 'Predicted winner'])
            df_test_proba = pd.DataFrame()

            # liste des courses par raceId
            round_list = list(np.sort(df_test['round'].unique()))
            
            # boucle sur chaque Grand Prix
            for n in round_list:
                # pour la 1ere course (round=1)
                #    jeux données train = jeux données initiales
                #    jeux données test = données de la course
                
                # pour les courses suivantes (round > 1)
                #    jeux données train = jeux données initiales + données des courses précédentes
                #    jeux données test = données de la course
                
                if n==1:
                    X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
                    y_train = df_train['positionOrder']
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']

                else:
                    X_previous_round = df_test[df_test['round']<=(n-1)].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_previous_round = df_test[df_test['round']<=(n-1)]['positionOrder']
                    
                    X_train = pd.concat([df_train.drop(['year', 'round', 'positionOrder'], axis=1), X_previous_round], axis=0)
                    y_train = pd.concat([df_train['positionOrder'], y_previous_round], axis=0)
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']

                
                # normalisation des données
                X_train_scaled = scaler.transform(X_train)
                X_test_round_n_scaled = scaler.transform(X_test_round_n)

                # rééchantillonnage
                X_ro, y_ro = ros.fit_resample(X_train_scaled, y_train)

                # entrainement du modèle
                log_reg.fit(X_ro, y_ro)

                # probabilité avec predict_proba
                y_pred_log_ro = log_reg.predict_proba(X_test_round_n_scaled)
                df_y_pred_log_ro = pd.DataFrame(y_pred_log_ro, columns=['proba_0', 'proba_1'])

                
                # dataframe des résultats de la course
                df_test_round_n_proba = pd.concat([df_test[df_test['round']==n].reset_index(), df_y_pred_log_ro], axis=1)
                # ajout colonne prediction initialisée à 0
                df_test_round_n_proba['prediction'] = 0

                # on identifie la valeur max de la probabilité classe 1 et on affecte valeur 1 dans prediction à l'index max
                max_proba_1 = df_test_round_n_proba['proba_1'].max()
                index_max_proba_1 = df_test_round_n_proba[df_test_round_n_proba['proba_1']==max_proba_1].index
                df_test_round_n_proba.loc[index_max_proba_1, 'prediction'] = 1
                
                
                # dataframe résultat global avec concaténation des données à chaque course
                if n==1:
                    df_test_proba = df_test_round_n_proba
                else:
                    df_test_proba = pd.concat([df_test_proba, df_test_round_n_proba], axis=0)

                # on identifie le pilote vainqueur réel
                real_winner = df_test_round_n_proba[df_test_round_n_proba['positionOrder']==1]['driverId'].values[0]
                # on identifie le pilote prédit vainqueur par le modèle
                predicted_winner = df_test_round_n_proba[df_test_round_n_proba['prediction']==1]['driverId'].values[0]
                
                # dataframe où on regroupe les vainqueurs réel et prédit de la course
                df_result_round_n = pd.DataFrame({'round' : [n],
                                                'Winner' : [real_winner],
                                                'Predicted winner' : [predicted_winner]})
                
                # on fusionne les vainqueurs dans le dataframe final
                df_winner = pd.concat([df_winner, df_result_round_n], axis=0)


            # rapport classification et matrice de confusion
            confusion_matrix_2 = pd.crosstab(df_test_proba['positionOrder'], df_test_proba['prediction'])
            confusion_matrix_2.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_2.index = ['Real val. 0', 'Real val. 1']

            classif_report_df_2 = pd.DataFrame(classification_report(df_test_proba['positionOrder'], df_test_proba['prediction'], output_dict=True)).T[:2]
            classif_report_df_2['support'] = classif_report_df_2['support'].astype('int')
            classif_report_df_2.index = ['Class 0', 'Class 1']
            
            col1_iter2, col2_iter2 = st.columns(2)
            with col1_iter2:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_df_2)

            # fusion des données pilotes dans le dataframe
            df_winner = df_winner.merge(right=drivers_data[['driverId', 'surname']], left_on='Winner', right_on='driverId')\
                                                .drop(['driverId', 'Winner'], axis=1)\
                                                .rename(columns={'surname' : 'Winner'})
            df_winner = df_winner.merge(right=drivers_data[['driverId', 'surname']], left_on='Predicted winner', right_on='driverId')\
                                                .drop(['driverId', 'Predicted winner'], axis=1)\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .sort_values(by=['round']).reset_index(drop=True)
            df_winner['match'] = df_winner.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)

            with col2_iter2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(df_winner, height=735)
    
    elif model_selector_2 == 'Forêt aléatoire':
        # ----------------------
        # Modèle Forêt aléatoire
        # ----------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter2, param_col2_iter2, param_col3_iter2, param_col4_iter2 = st.columns(4)

        with param_col1_iter2:
            n_estimators_param_selector = st.selectbox(label='n_estimators', options=(10, 50, 100, 250), index=2, key='rf_param1-iter2')
        with param_col2_iter2:
            min_samples_leaf_param_selector = st.selectbox(label='min_samples_leaf', options=(1, 3, 5), index=0, key='rf_param2-iter2')
        with param_col3_iter2:
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param3-iter2')

        if stateful_button(label='Résultats itération 2 - Forêt aléatoire', key='button-rf-iter2'):
        # if st.button('Résultats', key='rf-iter2'):

            st.write('---')

            # initialisation données features / target
            X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
            y_train = df_train['positionOrder']

            # instanciation fonction de normalisation des données
            scaler = StandardScaler().fit(X_train)
        
            # instanciation modèle
            rf = RandomForestClassifier(n_jobs=-1, max_features = max_features_param_selector, min_samples_leaf = min_samples_leaf_param_selector,
                                            n_estimators = n_estimators_param_selector, random_state=1430)
            
            # initialisation dataframe compilation des vainqueurs réels et prédits
            df_winner_rf = pd.DataFrame(columns=['round', 'Winner', 'Predicted winner'])
            df_test_proba_rf = pd.DataFrame()

            # liste des courses par raceId
            round_list_rf = list(np.sort(df_test['round'].unique()))


            # boucle sur chaque course
            for n in round_list_rf:
                # pour la 1ere course (round=1)
                #    jeux données train = jeux données initiales
                #    jeux données test = données de la course
                
                # pour les courses suivantes (round > 1)
                #    jeux données train = jeux données initiales + données des courses précédentes
                #    jeux données test = données de la course
                
                if n==1:
                    X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
                    y_train = df_train['positionOrder']
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']
                
                else:
                    X_previous_round = df_test[df_test['round']<=(n-1)].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_previous_round = df_test[df_test['round']<=(n-1)]['positionOrder']
                    
                    X_train = pd.concat([df_train.drop(['year', 'round', 'positionOrder'], axis=1), X_previous_round], axis=0)
                    y_train = pd.concat([df_train['positionOrder'], y_previous_round], axis=0)
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']

                    
                # normalisation des données
                X_train_scaled = scaler.transform(X_train)
                X_test_round_n_scaled = scaler.transform(X_test_round_n)

                # entrainement du modèle
                rf.fit(X_train_scaled, y_train)

                # probabilité avec predict_proba
                y_pred_rf_proba = rf.predict_proba(X_test_round_n_scaled)
                df_y_pred_rf_proba = pd.DataFrame(y_pred_rf_proba, columns=['proba_0', 'proba_1'])

                
                # dataframe des résultats de la course
                df_test_round_n_proba = pd.concat([df_test[df_test['round']==n].reset_index(), df_y_pred_rf_proba], axis=1)
                # ajout colonne prediction initialisée à 0
                df_test_round_n_proba['prediction'] = 0

                # on identifie la valeur max de la probabilité classe 1 et on affecte valeur 1 dans prediction à l'index max
                max_proba_1 = df_test_round_n_proba['proba_1'].max()
                index_max_proba_1 = df_test_round_n_proba[df_test_round_n_proba['proba_1']==max_proba_1].index
                df_test_round_n_proba.loc[index_max_proba_1, 'prediction'] = 1
                
                
                # dataframe résultat global avec concaténation des données à chaque course
                if n==1:
                    df_test_proba_rf = df_test_round_n_proba
                else:
                    df_test_proba_rf = pd.concat([df_test_proba_rf, df_test_round_n_proba], axis=0)

                # on identifie le pilote vainqueur réel
                real_winner = df_test_round_n_proba[df_test_round_n_proba['positionOrder']==1]['driverId'].values[0]
                # on identifie le pilote prédit vainqueur par le modèle
                predicted_winner = df_test_round_n_proba[df_test_round_n_proba['prediction']==1]['driverId'].values[0]
                
                # dataframe où on regroupe les vainqueurs réel et prédit de la course
                df_result_round_n = pd.DataFrame({'round' : [n],
                                                'Winner' : [real_winner],
                                                'Predicted winner' : [predicted_winner]})
                
                # on fusionne les vainqueurs dans le dataframe final
                df_winner_rf = pd.concat([df_winner_rf, df_result_round_n], axis=0)
            

            # rapport classification et matrice de confusion
            confusion_matrix_rf_2 = pd.crosstab(df_test_proba_rf['positionOrder'], df_test_proba_rf['prediction'])
            confusion_matrix_rf_2.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_rf_2.index = ['Real val. 0', 'Real val. 1']

            classif_report_rf_df_2 = pd.DataFrame(classification_report(df_test_proba_rf['positionOrder'], df_test_proba_rf['prediction'], output_dict=True)).T[:2]
            classif_report_rf_df_2['support'] = classif_report_rf_df_2['support'].astype('int')
            classif_report_rf_df_2.index = ['Class 0', 'Class 1']
            
            col1_iter2, col2_iter2 = st.columns(2)
            with col1_iter2:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_rf_2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_rf_df_2)

            # fusion des données pilotes dans le dataframe
            df_winner_rf = df_winner_rf.merge(right=drivers_data[['driverId', 'surname']], left_on='Winner', right_on='driverId')\
                                                .drop(['driverId', 'Winner'], axis=1)\
                                                .rename(columns={'surname' : 'Winner'})
            df_winner_rf = df_winner_rf.merge(right=drivers_data[['driverId', 'surname']], left_on='Predicted winner', right_on='driverId')\
                                                .drop(['driverId', 'Predicted winner'], axis=1)\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .sort_values(by=['round']).reset_index(drop=True)
            df_winner_rf['match'] = df_winner_rf.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)

            with col2_iter2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(df_winner_rf, height=735)
    
    elif model_selector_2 == 'Arbre de décision':
        # ------------------------
        # Modèle Arbre de décision
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter2, param_col2_iter2, param_col3_iter2, param_col4_iter2 = st.columns(4)

        with param_col1_iter2:
            criterion_param_selector = st.selectbox(label='criterion', options=('entropy', 'gini'), index=0, key='dt_param1-iter2')
        with param_col2_iter2:
            max_depth_param_selector = st.selectbox(label='max_depth', options=(1, 2, 3, 4, 5, 6, 7), index=4, key='dt_param2-iter2')

        if stateful_button(label='Résultats itération 2 - Arbre de décision', key='button-dt-iter2'):
        # if st.button('Résultats', key='dt-iter2'):

            st.write('---')

            # initialisation données features / target
            X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
            y_train = df_train['positionOrder']

            # instanciation fonction de normalisation des données
            scaler = StandardScaler().fit(X_train)

            # instanciation modèle
            dt_clf = DecisionTreeClassifier(criterion=criterion_param_selector, max_depth=max_depth_param_selector, random_state=143)
            
            # initialisation dataframe compilation des vainqueurs réels et prédits
            df_winner_dt = pd.DataFrame(columns=['round', 'Winner', 'Predicted winner'])
            df_test_proba_dt = pd.DataFrame()

            # liste des courses par raceId
            round_list_dt = list(np.sort(df_test['round'].unique()))

            # boucle sur chaque course
            for n in round_list_dt:
                # pour la 1ere course (round=1)
                #    jeux données train = jeux données initiales
                #    jeux données test = données de la course
                
                # pour les courses suivantes (round > 1)
                #    jeux données train = jeux données initiales + données des courses précédentes
                #    jeux données test = données de la course
                
                if n==1:
                    X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
                    y_train = df_train['positionOrder']
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']
                
                else:
                    X_previous_round = df_test[df_test['round']<=(n-1)].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_previous_round = df_test[df_test['round']<=(n-1)]['positionOrder']
                    
                    X_train = pd.concat([df_train.drop(['year', 'round', 'positionOrder'], axis=1), X_previous_round], axis=0)
                    y_train = pd.concat([df_train['positionOrder'], y_previous_round], axis=0)
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']

                    
                # normalisation des données
                X_train_scaled = scaler.transform(X_train)
                X_test_round_n_scaled = scaler.transform(X_test_round_n)

                # rééchantillonnage
                X_ro, y_ro = ros.fit_resample(X_train_scaled, y_train)

                # entrainement du modèle
                dt_clf.fit(X_ro, y_ro)

                # probabilité avec predict_proba
                y_pred_dt_proba = dt_clf.predict_proba(X_test_round_n_scaled)
                df_y_pred_dt_proba = pd.DataFrame(y_pred_dt_proba, columns=['proba_0', 'proba_1'])

                
                # dataframe des résultats de la course
                df_test_round_n_proba = pd.concat([df_test[df_test['round']==n].reset_index(), df_y_pred_dt_proba], axis=1)
                # ajout colonne prediction initialisée à 0
                df_test_round_n_proba['prediction'] = 0

                # on identifie la valeur max de la probabilité classe 1 et on affecte valeur 1 dans prediction à l'index max
                max_proba_1 = df_test_round_n_proba['proba_1'].max()
                index_max_proba_1 = df_test_round_n_proba[df_test_round_n_proba['proba_1']==max_proba_1].index
                df_test_round_n_proba.loc[index_max_proba_1, 'prediction'] = 1
                
                
                # dataframe résultat global avec concaténation des données à chaque course
                if n==1:
                    df_test_proba_dt = df_test_round_n_proba
                else:
                    df_test_proba_dt = pd.concat([df_test_proba_dt, df_test_round_n_proba], axis=0)

                # on identifie le pilote vainqueur réel
                real_winner = df_test_round_n_proba[df_test_round_n_proba['positionOrder']==1]['driverId'].values[0]
                # on identifie le pilote prédit vainqueur par le modèle
                predicted_winner = df_test_round_n_proba[df_test_round_n_proba['prediction']==1]['driverId'].values[0]
                
                # dataframe où on regroupe les vainqueurs réel et prédit de la course
                df_result_round_n = pd.DataFrame({'round' : [n],
                                                'Winner' : [real_winner],
                                                'Predicted winner' : [predicted_winner]})
                
                # on fusionne les vainqueurs dans le dataframe final
                df_winner_dt = pd.concat([df_winner_dt, df_result_round_n], axis=0)
            
            # rapport classification et matrice de confusion
            confusion_matrix_dt_2 = pd.crosstab(df_test_proba_dt['positionOrder'], df_test_proba_dt['prediction'])
            confusion_matrix_dt_2.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_dt_2.index = ['Real val. 0', 'Real val. 1']

            classif_report_dt_df_2 = pd.DataFrame(classification_report(df_test_proba_dt['positionOrder'], df_test_proba_dt['prediction'], output_dict=True)).T[:2]
            classif_report_dt_df_2['support'] = classif_report_dt_df_2['support'].astype('int')
            classif_report_dt_df_2.index = ['Class 0', 'Class 1']
            
            col1_iter2, col2_iter2 = st.columns(2)
            with col1_iter2:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_dt_2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_dt_df_2)


            df_winner_dt = df_winner_dt.merge(right=drivers_data[['driverId', 'surname']], left_on='Winner', right_on='driverId')\
                                                .drop(['driverId', 'Winner'], axis=1)\
                                                .rename(columns={'surname' : 'Winner'})
            df_winner_dt = df_winner_dt.merge(right=drivers_data[['driverId', 'surname']], left_on='Predicted winner', right_on='driverId')\
                                                .drop(['driverId', 'Predicted winner'], axis=1)\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .sort_values(by=['round']).reset_index(drop=True)
            df_winner_dt['match'] = df_winner_dt.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)

            with col2_iter2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(df_winner_dt, height=735)
   
    elif model_selector_2 == 'KNN':
        # ------------------------
        # Modèle KNN
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter2, param_col2_iter2, param_col3_iter2, param_col4_iter2 = st.columns(4)

        with param_col1_iter2:
            n_neighbors_param_selector = st.selectbox(label='n_neighbors', options=(2, 5, 7, 10), index=3, key='knn_param1-iter2')
        with param_col2_iter2:
            metric_param_selector = st.selectbox(label='metric', options=('minkowski', 'manhattan', 'chebyshev'), index=1, key='knn_param2-iter2')

        if stateful_button(label='Résultats itération 2 - KNN', key='button-knn-iter2'):
        # if st.button('Résultats', key='knn-iter2'):

            st.write('---')

            # initialisation données features / target
            X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
            y_train = df_train['positionOrder']

            # instanciation fonction de normalisation des données
            scaler = StandardScaler().fit(X_train)

            # instanciation modèle
            knn_ro = KNeighborsClassifier(n_neighbors=n_neighbors_param_selector, metric=metric_param_selector)
            
            # initialisation dataframe compilation des vainqueurs réels et prédits
            df_winner_knn_ro = pd.DataFrame(columns=['round', 'Winner', 'Predicted winner'])
            df_test_proba_knn_ro = pd.DataFrame()

            # liste des courses par raceId
            round_list_knn_ro = list(np.sort(df_test['round'].unique()))

            # boucle sur chaque course
            for n in round_list_knn_ro:
                # pour la 1ere course (round=1)
                #    jeux données train = jeux données initiales
                #    jeux données test = données de la course
                
                # pour les courses suivantes (round > 1)
                #    jeux données train = jeux données initiales + données des courses précédentes
                #    jeux données test = données de la course
                
                if n==1:
                    X_train = df_train.drop(['year', 'round', 'positionOrder'], axis=1)
                    y_train = df_train['positionOrder']
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']
                
                else:
                    X_previous_round = df_test[df_test['round']<=(n-1)].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_previous_round = df_test[df_test['round']<=(n-1)]['positionOrder']
                    
                    X_train = pd.concat([df_train.drop(['year', 'round', 'positionOrder'], axis=1), X_previous_round], axis=0)
                    y_train = pd.concat([df_train['positionOrder'], y_previous_round], axis=0)
                    
                    X_test_round_n = df_test[df_test['round']==n].drop(['year', 'round', 'positionOrder'], axis=1)
                    y_test_round_n = df_test[df_test['round']==n]['positionOrder']

                # normalisation des données
                X_train_scaled = scaler.transform(X_train)
                X_test_round_n_scaled = scaler.transform(X_test_round_n)

                # rééchantillonnage
                X_ro, y_ro = ros.fit_resample(X_train_scaled, y_train)

                # entrainement du modèle
                knn_ro.fit(X_ro,y_ro)

                # probabilité avec predict_proba
                y_pred_knn_ro_proba = knn_ro.predict_proba(X_test_round_n_scaled)
                df_y_pred_knn_ro_proba = pd.DataFrame(y_pred_knn_ro_proba, columns=['proba_0', 'proba_1'])

                # dataframe des résultats de la course
                df_test_round_n_proba = pd.concat([df_test[df_test['round']==n].reset_index(), df_y_pred_knn_ro_proba], axis=1)
                # ajout colonne prediction initialisée à 0
                df_test_round_n_proba['prediction'] = 0

                # on identifie la valeur max de la probabilité classe 1 et on affecte valeur 1 dans prediction à l'index max
                max_proba_1 = df_test_round_n_proba['proba_1'].max()
                index_max_proba_1 = df_test_round_n_proba[df_test_round_n_proba['proba_1']==max_proba_1].index
                df_test_round_n_proba.loc[index_max_proba_1, 'prediction'] = 1

                # dataframe résultat global avec concaténation des données à chaque course
                if n==1:
                    df_test_proba_knn_ro = df_test_round_n_proba
                else:
                    df_test_proba_knn_ro = pd.concat([df_test_proba_knn_ro, df_test_round_n_proba], axis=0)

                # on identifie le pilote vainqueur réel
                real_winner = df_test_round_n_proba[df_test_round_n_proba['positionOrder']==1]['driverId'].values[0]
                # on identifie le pilote prédit vainqueur par le modèle
                predicted_winner = df_test_round_n_proba[df_test_round_n_proba['prediction']==1]['driverId'].values[0]
                
                # dataframe où on regroupe les vainqueurs réel et prédit de la course
                df_result_round_n = pd.DataFrame({'round' : [n],
                                                'Winner' : [real_winner],
                                                'Predicted winner' : [predicted_winner]})
                
                # on fusionne les vainqueurs dans le dataframe final
                df_winner_knn_ro = pd.concat([df_winner_knn_ro, df_result_round_n], axis=0)


            # rapport classification et matrice de confusion
            confusion_matrix_knn_ro_2 = pd.crosstab(df_test_proba_knn_ro['positionOrder'], df_test_proba_knn_ro['prediction'])
            confusion_matrix_knn_ro_2.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_knn_ro_2.index = ['Real val. 0', 'Real val. 1']

            classif_report_knn_ro_df_2 = pd.DataFrame(classification_report(df_test_proba_knn_ro['positionOrder'], df_test_proba_knn_ro['prediction'], output_dict=True)).T[:2]
            classif_report_knn_ro_df_2['support'] = classif_report_knn_ro_df_2['support'].astype('int')
            classif_report_knn_ro_df_2.index = ['Class 0', 'Class 1']
            
            col1_iter2, col2_iter2 = st.columns(2)
            with col1_iter2:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_knn_ro_2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_knn_ro_df_2)


            df_winner_knn_ro = df_winner_knn_ro.merge(right=drivers_data[['driverId', 'surname']], left_on='Winner', right_on='driverId')\
                                                .drop(['driverId', 'Winner'], axis=1)\
                                                .rename(columns={'surname' : 'Winner'})
            df_winner_knn_ro = df_winner_knn_ro.merge(right=drivers_data[['driverId', 'surname']], left_on='Predicted winner', right_on='driverId')\
                                                .drop(['driverId', 'Predicted winner'], axis=1)\
                                                .rename(columns={'surname' : 'Predicted winner'})\
                                                .sort_values(by=['round']).reset_index(drop=True)
            df_winner_knn_ro['match'] = df_winner_knn_ro.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)

            with col2_iter2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(df_winner_knn_ro, height=735)
    

    
    # ----------------------------
    # Conclusion
    # ----------------------------
    st.write('---')
    st.markdown(
        """
        ## Résultats

        """)
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.facecolor'] = '#0e1117'
    plt.rcParams['axes.edgecolor'] = '#38383f'
    plt.rcParams['axes.titlecolor'] = '#fff'
    plt.rcParams['axes.labelcolor'] = '#fff'
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.grid.axis'] = 'y'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['grid.color'] = '#d0d0d2'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['xtick.color'] = '#fff'
    plt.rcParams['ytick.color'] = '#fff'
    plt.rcParams['figure.facecolor'] = '#15151e'

    models_iter1 = ['Régression\nlogistique', 'Foret\naléatoire', 'Arbre de\ndécision', 'SVC', 'KNN']
    score_models_iter1  = [0.55, 0.33, 0.50, 0.50, 0.29]
    color_models_iter1  = [ '#e10600', '#4c78a8', '#4c78a8', '#4c78a8', '#4c78a8']

    fig_iter1 = plt.figure(figsize=(6.5, 3.5))
    plt.bar(x=models_iter1, height=score_models_iter1, width=0.6, color=color_models_iter1)
    plt.title('Modèles')
    plt.ylabel('Score')
    plt.ylim([0,1])

    for i in range(len(score_models_iter1)):
        plt.annotate(str(score_models_iter1[i]), xy=(models_iter1[i], score_models_iter1[i]), ha='center', va='bottom', color='#fff')
    

    models_iter2 = ['Régression\nlogistique', 'Foret\naléatoire', 'Arbre de\ndécision', 'KNN']
    score_models_iter2  = [0.55, 0.45, 0.50, 0.31]
    color_models_iter2  = [ '#e10600', '#4c78a8', '#4c78a8', '#4c78a8', '#4c78a8']

    fig_iter2 = plt.figure(figsize=(6.5, 3.5))
    plt.bar(x=models_iter2, height=score_models_iter2, width=0.6, color=color_models_iter2)
    plt.title('Modèles')
    plt.ylabel('Score')
    plt.ylim([0,1])

    for i in range(len(score_models_iter2)):
        plt.annotate(str(score_models_iter2[i]), xy=(models_iter2[i], score_models_iter2[i]), ha='center', va='bottom', color='#fff')

    col1_results, col2_results = st.columns(2)
    with col1_results:
        st.write(
            """
            #### Itération 1
            """)
        st.pyplot(fig_iter1)
    
    with col2_results:
        st.write(
            """
            #### Itération 2
            """)
        st.pyplot(fig_iter2)
    
    st.markdown(
        """
        Les modèles obtiennent sensiblement les mêmes scores, mis à part le modèle de **foret aléatoire** où on observe une amélioration significative dans la 2e itération.

        Il est également à noter que le modèle **SVC** a un temps de calcul assez long. Il n'a pas été retenu pour la 2e itération, vu qu'une boucle est appliquée à chaque Grand Prix de la saison.

        Avec un score de 55%, le modèle **régression logistique** obtient un bon résultat. Le modèle semble relativement bon pour prédire les favoris mais n’arrive pas à trouver les « _outsiders_ » (cote entre 2.1 et 4). On observe la même tendance pour les autres modèles avec un score de 50%.

        On peut donc voir que la qualité d’un modèle se dessine plutôt sur les « _outsiders_ ». Il semble aisé pour un modèle de prédire un favori, mais la différence entre deux modèles se fera surtout sur les « _outsiders_ », voire les « _upsets_ » (pilotes qui bouleversent les statistiques et avec une cote > 4).

        Nous avons simulé les paris sur le Championnat 2021 avec les meilleurs résultats obtenus de chaque modèle, en misant 100 € sur un pilote (mise totale 2000 € sur la saison).
        
        Nous aurions obtenu les montants ci-dessous :

        """, unsafe_allow_html=True)
    
    st.table(pd.DataFrame({'Modèle' : ['Régression logistique', 'Arbre de décision', 'KNN', 'Forêt aléatoire'],
                            'Résultat net' : ['320 €', '50 €', '-20 €', '-210 €']}))
    
    st.markdown(
        """
        Sur les quatre modèles, nous avons été bénéficiaires sur deux modèles. La régression logistique se détache clairement puisqu’elle a permis un bénéfice de 320€ pour une mise de 2000€, soit un ROI de 16%.

        La différence entre les modèles, sur le plan financier, se fait bien sur ces « _outsiders_ ».
        """)