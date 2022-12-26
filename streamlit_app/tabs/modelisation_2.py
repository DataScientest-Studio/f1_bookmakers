import streamlit as st

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

title = "Modélisations"
sidebar_name = "Modélisation - Top 3"

def run():

    st.markdown('<style>section[tabindex="0"] div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child {flex-basis: 30%;}</style>', unsafe_allow_html=True)

    def df_background_color(s):
        return ['background-color: #202028']*len(s) if (s['round']%2)==0 else ['background-color: #0e1117']*len(s)

    st.title(title)

    # -----------------------
    # préparation des données
    # -----------------------

    st.markdown('## Données')

    # chargement données
    df = pd.read_csv(r"../data/df_results_meteo_circuit_classement.csv", sep=';', index_col=0)
    # chargement données pilotes
    drivers_data = pd.read_csv(r"../data/drivers.csv")

    # suppression des lignes avec nan
    df = df.dropna()

    df_columns = ['driverId', 'constructorId', 'grid', 'fastestLapSpeed_classes', 'positionOrder', 'driverStandingPosition', 'driverWins', 'constructorStandingPosition', 'constructorWins']
    df[df_columns] = df[df_columns].astype('int')

    # ajout variable 'podium' avec valeur 1 pour positionOrder = 1/2/3 sinon 0 pour les autres valeurs
    df['podium'] = df['positionOrder'].apply(lambda x: 0 if x>3 else 1)
    # modif valeurs positionOrder 1/2/3 pour prédire top 3 sinon 0 pour les autres valeurs
    df['positionOrder'] = df['positionOrder'].apply(lambda x: 0 if x>3 else x)
    st.dataframe(df.head(20))

    # jeux données train/test
    df_train = df[df['year']<=2020]
    df_test = df[df['year']==2021]

    # séparation données features / target
    X_train = df_train.drop(['year', 'round', 'positionOrder', 'podium'], axis=1)
    y_train_podium = df_train['podium']
    y_train_top3 = df_train['positionOrder']

    X_test = df_test.drop(['year', 'round', 'positionOrder', 'podium'], axis=1)
    y_test_podium = df_test['podium']
    y_test_top3 = df_test['positionOrder']

    # normalisation des données
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # rééchantillonnage (Régression Log + Arbre de décision)
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    X_ro_top3, y_ro_top3 = ros.fit_resample(X_train_scaled, y_train_top3)
    X_ro_podium, y_ro_podium = ros.fit_resample(X_train_scaled, y_train_podium)
    
    # ----------------------------
    # Algorithmes
    # ----------------------------
    st.markdown(
        """
        ## Top 3

        Nous souhaitons explorer la possibilité de prédire le top 3 d’arrivée d’une course.

        Nous ajustons la variable cible « positionOrder » en gardant les uniquement les positions 1 / 2 / 3, les autres valeurs sont mises à zéro.

        """)
    
    model_selector = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision'), key="iter1",
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)
    
    if model_selector == 'Régression logistique':
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter1, param_col2_iter1, param_col3_iter1, param_col4_iter1 = st.columns(4)

        with param_col1_iter1:
            C_param_selector = st.selectbox(label='C', options=(0.001, 0.01, 0.1, 1, 10), index=0, key='log-iter1')

        if st.button('Résultats', key='log-iter1'):  

            st.write('---')

            # instanciation modèle
            log_reg = LogisticRegression(C=C_param_selector)
            log_reg.fit(X_ro_top3, y_ro_top3)

            # probabilité avec predict_proba
            y_pred_log_ro = log_reg.predict_proba(X_test_scaled)
            df_y_pred_log_ro = pd.DataFrame(y_pred_log_ro, columns=['proba_0', 'proba_1', 'proba_2', 'proba_3'])

            # création dataframe des résultats
            df_test_proba = pd.concat([df_test.reset_index(), df_y_pred_log_ro], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_proba['prediction'] = 0

            # liste des courses par raceId
            raceId_list = df_test_proba['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list:
                df_temp = df_test_proba[df_test_proba['raceId']==i]                 # filtre les données de la course
                
                max_proba_1 = df_temp['proba_1'].max()                              # on identifie la valeur max de la probabilité classe 1
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index  # on récupère l'index de la valeur max
                df_test_proba.loc[index_max_proba_1, 'prediction'] = 1              # on affecte valeur 1 dans prediction à l'index max
                df_temp.loc[index_max_proba_1, 'prediction'] = 1
                
                max_proba_2 = df_temp[df_temp['prediction']==0]['proba_2'].max()    # on identifie la valeur max proba 2 en filtrant la ligne max proba 1
                index_max_proba_2 = df_temp[df_temp['proba_2']==max_proba_2].index  # on récupère l'index de la valeur max
                df_test_proba.loc[index_max_proba_2, 'prediction'] = 2              # on affecte valeur 2 dans prediction à l'index max
                df_temp.loc[index_max_proba_2, 'prediction'] = 2
                
                max_proba_3 = df_temp[df_temp['prediction']==0]['proba_3'].max()    # on identifie la valeur max proba 3 en filtrant la ligne max proba 1 et 2
                index_max_proba_3 = df_temp[df_temp['proba_3']==max_proba_3].index  # on récupère l'index de la valeur max
                df_test_proba.loc[index_max_proba_3, 'prediction'] = 3              # on affecte valeur 3 dans prediction à l'index max
            
            # rapport classification et matrice de confusion
            confusion_matrix = pd.crosstab(df_test_proba['positionOrder'], df_test_proba['prediction'])
            confusion_matrix.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_df = pd.DataFrame(classification_report(df_test_proba['positionOrder'], df_test_proba['prediction'], output_dict=True)).T[:4]
            classif_report_df['support'] = classif_report_df['support'].astype('int')
            classif_report_df.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_df)
            
            # dataframe des vainqueurs réels
            podium_real_log = df_test_proba[df_test_proba["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_log = podium_real_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des vainqueurs prédits
            podium_predicted_log = df_test_proba[df_test_proba["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_log = podium_predicted_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_log = podium_real_log.merge(right=podium_predicted_log, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_log['Top 3 ranked'] = podium_real_log.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list = podium_real_log['round'].unique()
            podium_real_log['Top 3 unranked'] = '❌'

            for i in rounds_list:
                temp_df = podium_real_log[podium_real_log['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_log.loc[index_driver, 'Top 3 unranked'] = '✔'

            with col2_iter1:
                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_log.style.apply(df_background_color, axis=1), height=735)
    

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
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param2-iter1')

        if st.button('Résultats', key='rf-iter1'):

            st.write('---')
        
            # instanciation modèle
            rf = RandomForestClassifier(n_jobs=-1, max_features = max_features_param_selector, min_samples_leaf = min_samples_leaf_param_selector,
                                            n_estimators = n_estimators_param_selector, random_state=1430)
            rf.fit(X_train_scaled, y_train_top3)

            # probabilité avec predict_proba
            y_pred_rf_proba = rf.predict_proba(X_test_scaled)
            df_y_pred_rf_proba = pd.DataFrame(y_pred_rf_proba, columns=['proba_0', 'proba_1', 'proba_2', 'proba_3'])

            # création dataframe des résultats
            df_test_prob_rf = pd.concat([df_test.reset_index(), df_y_pred_rf_proba], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_rf['prediction'] = 0

            # liste des courses par raceId
            raceId_listRF = df_test_prob_rf['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_listRF:
                df_temp = df_test_prob_rf[df_test_prob_rf['raceId']==i]             # filtre les données de la course
    
                max_proba_1 = df_temp['proba_1'].max()                              # on identifie la valeur max de la probabilité classe 1
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index  # on récupère l'index de la valeur max
                df_test_prob_rf.loc[index_max_proba_1, 'prediction'] = 1            # on affecte valeur 1 dans prediction à l'index max
                df_temp.loc[index_max_proba_1, 'prediction'] = 1
                
                max_proba_2 = df_temp[df_temp['prediction']==0]['proba_2'].max()    # on identifie la valeur max proba 2 en filtrant la ligne max proba 1
                index_max_proba_2 = df_temp[df_temp['proba_2']==max_proba_2].index  # on récupère l'index de la valeur max
                df_test_prob_rf.loc[index_max_proba_2, 'prediction'] = 2            # on affecte valeur 2 dans prediction à l'index max
                df_temp.loc[index_max_proba_2, 'prediction'] = 2
                
                max_proba_3 = df_temp[df_temp['prediction']==0]['proba_3'].max()    # on identifie la valeur max proba 3 en filtrant la ligne max proba 1 et 2
                index_max_proba_3 = df_temp[df_temp['proba_3']==max_proba_3].index  # on récupère l'index de la valeur max
                df_test_prob_rf.loc[index_max_proba_3, 'prediction'] = 3            # on affecte valeur 3 dans prediction à l'index max
            

            # rapport classification et matrice de confusion
            confusion_matrix_rf = pd.crosstab(df_test_prob_rf['positionOrder'], df_test_prob_rf['prediction'])
            confusion_matrix_rf.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_rf.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_rf_df = pd.DataFrame(classification_report(df_test_prob_rf['positionOrder'], df_test_prob_rf['prediction'], output_dict=True)).T[:4]
            classif_report_rf_df['support'] = classif_report_rf_df['support'].astype('int')
            classif_report_rf_df.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_rf)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_rf_df)


            # dataframe des vainqueurs réels
            podium_real_rf = df_test_prob_rf[df_test_prob_rf["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_rf = podium_real_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des vainqueurs prédits
            podium_predicted_rf = df_test_prob_rf[df_test_prob_rf["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_rf = podium_predicted_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_rf = podium_real_rf.merge(right=podium_predicted_rf, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_rf['Top 3 ranked'] = podium_real_rf.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list = podium_real_rf['round'].unique()
            podium_real_rf['Top 3 unranked'] = '❌'

            for i in rounds_list:
                temp_df = podium_real_rf[podium_real_rf['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_rf.loc[index_driver, 'Top 3 unranked'] = '✔'

            with col2_iter1:
                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_rf.style.apply(df_background_color, axis=1), height=735)
    

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

        if st.button('Résultats', key='dt-iter1'):

            st.write('---')

            # instanciation modèle
            dt_clf = DecisionTreeClassifier(criterion=criterion_param_selector, max_depth=max_depth_param_selector, random_state=143)
            dt_clf.fit(X_ro_top3, y_ro_top3)

            # probabilité avec predict_proba
            y_pred_dt_proba = dt_clf.predict_proba(X_test_scaled)
            df_y_pred_dt_proba = pd.DataFrame(y_pred_dt_proba, columns=['proba_0', 'proba_1', 'proba_2', 'proba_3'])


            # création dataframe des résultats
            df_test_prob_dt = pd.concat([df_test.reset_index(), df_y_pred_dt_proba], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_dt['prediction'] = 0


            # liste des courses par raceId
            raceId_list_dt = df_test_prob_dt['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list_dt:
                df_temp = df_test_prob_dt[df_test_prob_dt['raceId']==i]             # filtre les données de la course
    
                max_proba_1 = df_temp['proba_1'].max()                              # on identifie la valeur max de la probabilité classe 1
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index  # on récupère l'index de la valeur max
                df_test_prob_dt.loc[index_max_proba_1, 'prediction'] = 1            # on affecte valeur 1 dans prediction à l'index max
                df_temp.loc[index_max_proba_1, 'prediction'] = 1
                
                max_proba_2 = df_temp[df_temp['prediction']==0]['proba_2'].max()    # on identifie la valeur max proba 2 en filtrant la ligne max proba 1
                index_max_proba_2 = df_temp[df_temp['proba_2']==max_proba_2].index  # on récupère l'index de la valeur max
                df_test_prob_dt.loc[index_max_proba_2, 'prediction'] = 2            # on affecte valeur 2 dans prediction à l'index max
                df_temp.loc[index_max_proba_2, 'prediction'] = 2
                
                max_proba_3 = df_temp[df_temp['prediction']==0]['proba_3'].max()    # on identifie la valeur max proba 3 en filtrant la ligne max proba 1 et 2
                index_max_proba_3 = df_temp[df_temp['proba_3']==max_proba_3].index  # on récupère l'index de la valeur max
                df_test_prob_dt.loc[index_max_proba_3, 'prediction'] = 3            # on affecte valeur 3 dans prediction à l'index max
            

            # rapport classification et matrice de confusion
            confusion_matrix_dt = pd.crosstab(df_test_prob_dt['positionOrder'], df_test_prob_dt['prediction'])
            confusion_matrix_dt.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_dt.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_dt_df = pd.DataFrame(classification_report(df_test_prob_dt['positionOrder'], df_test_prob_dt['prediction'], output_dict=True)).T[:4]
            classif_report_dt_df['support'] = classif_report_dt_df['support'].astype('int')
            classif_report_dt_df.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
            
            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_dt)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_dt_df)


            # dataframe des vainqueurs réels
            podium_real_dt = df_test_prob_dt[df_test_prob_dt["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_dt = podium_real_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des vainqueurs prédits
            podium_predicted_dt = df_test_prob_dt[df_test_prob_dt["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_dt = podium_predicted_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_dt = podium_real_dt.merge(right=podium_predicted_dt, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_dt['Top 3 ranked'] = podium_real_dt.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list = podium_real_dt['round'].unique()
            podium_real_dt['Top 3 unranked'] = '❌'

            for i in rounds_list:
                temp_df = podium_real_dt[podium_real_dt['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_dt.loc[index_driver, 'Top 3 unranked'] = '✔'

            with col2_iter1:
                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_dt.style.apply(df_background_color, axis=1), height=735)
    
    st.write('---')

    st.markdown(
        """
        ## Podium

        Nous avons opté pour une variable cible « podium » qui a pour valeur 1 les positions 1 / 2 / 3 de la variable « positionOrder » et zéro pour les autres positions.

        """)
    
    model_selector_2 = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision'), key='iter2',
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)

    if model_selector_2 == 'Régression logistique':
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter2, param_col2_iter2, param_col3_iter2, param_col4_iter2 = st.columns(4)

        with param_col1_iter2:
            C_param_selector = st.selectbox(label='C', options=(0.001, 0.01, 0.1, 1, 10), index=0, key='log-iter2')

        if st.button('Résultats', key='log-iter2'):  

            st.write('---')

            # instanciation modèle
            log_reg = LogisticRegression(C=C_param_selector)
            log_reg.fit(X_ro_podium, y_ro_podium)

            # probabilité avec predict_proba
            y_pred_log_ro = log_reg.predict_proba(X_test_scaled)
            df_y_pred_log_ro = pd.DataFrame(y_pred_log_ro, columns=['proba_0', 'proba_1'])

            # création dataframe des résultats
            df_test_proba = pd.concat([df_test.reset_index(), df_y_pred_log_ro], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_proba['prediction'] = 0

            # liste des courses par raceId
            raceId_list = df_test_proba['raceId'].unique()

            
            # boucle sur chaque course
            for i in raceId_list:
                df_temp = df_test_proba[df_test_proba['raceId']==i]                             # filtre les données de la course
                index_podium = df_temp.loc[:,'proba_1'].sort_values(ascending=False)[:3].index  # on récupère l'index des 3 valeurs max
                df_test_proba.loc[index_podium, 'prediction'] = 1                               # on affecte valeur 1 dans prediction aux 3 index max


            # rapport classification et matrice de confusion
            confusion_matrix_2 = pd.crosstab(df_test_proba['podium'], df_test_proba['prediction'])
            confusion_matrix_2.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_2.index = ['Real val. 0', 'Real val. 1']

            classif_report_df_2 = pd.DataFrame(classification_report(df_test_proba['podium'], df_test_proba['prediction'], output_dict=True)).T[:2]
            classif_report_df_2['support'] = classif_report_df_2['support'].astype('int')
            classif_report_df_2.index = ['Class 0', 'Class 1']
            
            col1_iter2, col2_iter2 = st.columns(2)
            with col1_iter2:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_df_2)

            # dataframe des vainqueurs réels
            podium_real_log = df_test_proba[df_test_proba["podium"]==1][['round', 'driverId']]
            podium_real_log = podium_real_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .sort_values(by=['round'])\
                                                        .drop(['driverId'], axis=1)\
                                                        .reset_index(drop=True)

            # dataframe des vainqueurs prédits
            podium_predicted_log = df_test_proba[df_test_proba["prediction"]==1][['round', 'driverId']]
            podium_predicted_log = podium_predicted_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .sort_values(by=['round'])\
                                                            .drop(['driverId'], axis=1)\
                                                            .reset_index(drop=True)

            # fusion des 2 dataframes
            podium_real_log = pd.concat(objs=[podium_real_log, podium_predicted_log['Predicted driver']], axis=1)\

            rounds_list = podium_real_log['round'].unique()
            podium_real_log['on podium'] = '❌'

            for i in rounds_list:
                temp_df = podium_real_log[podium_real_log['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_log.loc[index_driver, 'on podium'] = '✅'

            with col2_iter2:
                st.markdown("""#### Pilotes podium VS prédictions""")
                st._legacy_dataframe(podium_real_log.style.apply(df_background_color, axis=1), height=735)
    

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
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param2-iter2')

        if st.button('Résultats', key='rf-iter2'):

            st.write('---')

            # instanciation modèle
            rf = RandomForestClassifier(n_jobs=-1, max_features = max_features_param_selector, min_samples_leaf = min_samples_leaf_param_selector,
                                            n_estimators = n_estimators_param_selector, random_state=1430)
            rf.fit(X_train_scaled, y_train_podium)

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
                df_temp = df_test_prob_rf[df_test_prob_rf['raceId']==i]                         # filtre les données de la course
                index_podium = df_temp.loc[:,'proba_1'].sort_values(ascending=False)[:3].index  # on récupère l'index des 3 valeurs max
                df_test_prob_rf.loc[index_podium, 'prediction'] = 1                             # on affecte valeur 1 dans prediction aux 3 index max
            
            # rapport classification et matrice de confusion
            confusion_matrix_rf_2 = pd.crosstab(df_test_prob_rf['podium'], df_test_prob_rf['prediction'])
            confusion_matrix_rf_2.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_rf_2.index = ['Real val. 0', 'Real val. 1']

            classif_report_rf_df_2 = pd.DataFrame(classification_report(df_test_prob_rf['podium'], df_test_prob_rf['prediction'], output_dict=True)).T[:2]
            classif_report_rf_df_2['support'] = classif_report_rf_df_2['support'].astype('int')
            classif_report_rf_df_2.index = ['Class 0', 'Class 1']
            
            col1_iter2, col2_iter2 = st.columns(2)
            with col1_iter2:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_rf_2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_rf_df_2)
            

            # dataframe des vainqueurs réels
            podium_real_rf = df_test_prob_rf[df_test_prob_rf["podium"]==1][['round', 'driverId']]
            podium_real_rf = podium_real_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .sort_values(by=['round'])\
                                                        .drop(['driverId'], axis=1)\
                                                        .reset_index(drop=True)

            # dataframe des vainqueurs prédits
            podium_predicted_rf = df_test_prob_rf[df_test_prob_rf["prediction"]==1][['round', 'driverId']]
            podium_predicted_rf = podium_predicted_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .sort_values(by=['round'])\
                                                            .drop(['driverId'], axis=1)\
                                                            .reset_index(drop=True)

            # fusion des 2 dataframes
            podium_real_rf = pd.concat(objs=[podium_real_rf, podium_predicted_rf['Predicted driver']], axis=1)\

            rounds_list = podium_real_rf['round'].unique()
            podium_real_rf['on podium'] = '❌'

            for i in rounds_list:
                temp_df = podium_real_rf[podium_real_rf['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_rf.loc[index_driver, 'on podium'] = '✅'

            with col2_iter2:
                st.markdown("""#### Pilotes podium VS prédictions""")
                st._legacy_dataframe(podium_real_rf.style.apply(df_background_color, axis=1), height=735)

    
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

        if st.button('Résultats', key='dt-iter2'):

            st.write('---')

            # instanciation modèle
            dt_clf = DecisionTreeClassifier(criterion=criterion_param_selector, max_depth=max_depth_param_selector, random_state=143)
            dt_clf.fit(X_ro_podium, y_ro_podium)

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
                df_temp = df_test_prob_dt[df_test_prob_dt['raceId']==i]                         # filtre les données de la course
                index_podium = df_temp.loc[:,'proba_1'].sort_values(ascending=False)[:3].index  # on récupère l'index des 3 valeurs max
                df_test_prob_dt.loc[index_podium, 'prediction'] = 1                             # on affecte valeur 1 dans prediction aux 3 index max
            
            # rapport classification et matrice de confusion
            confusion_matrix_dt_2 = pd.crosstab(df_test_prob_dt['podium'], df_test_prob_dt['prediction'])
            confusion_matrix_dt_2.columns = ['Pred. 0', 'Pred. 1']
            confusion_matrix_dt_2.index = ['Real val. 0', 'Real val. 1']

            classif_report_dt_df_2 = pd.DataFrame(classification_report(df_test_prob_dt['podium'], df_test_prob_dt['prediction'], output_dict=True)).T[:2]
            classif_report_dt_df_2['support'] = classif_report_dt_df_2['support'].astype('int')
            classif_report_dt_df_2.index = ['Class 0', 'Class 1']

            col1_iter2, col2_iter2 = st.columns(2)
            with col1_iter2:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_dt_2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_dt_df_2)
            

            # dataframe des vainqueurs réels
            podium_real_dt = df_test_prob_dt[df_test_prob_dt["podium"]==1][['round', 'driverId']]
            podium_real_dt = podium_real_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .sort_values(by=['round'])\
                                                        .drop(['driverId'], axis=1)\
                                                        .reset_index(drop=True)

            # dataframe des vainqueurs prédits
            podium_predicted_dt = df_test_prob_dt[df_test_prob_dt["prediction"]==1][['round', 'driverId']]
            podium_predicted_dt = podium_predicted_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .sort_values(by=['round'])\
                                                            .drop(['driverId'], axis=1)\
                                                            .reset_index(drop=True)

            # fusion des 2 dataframes
            podium_real_dt = pd.concat(objs=[podium_real_dt, podium_predicted_dt['Predicted driver']], axis=1)\

            rounds_list = podium_real_dt['round'].unique()
            podium_real_dt['on podium'] = '❌'

            for i in rounds_list:
                temp_df = podium_real_dt[podium_real_dt['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_dt.loc[index_driver, 'on podium'] = '✅'

            with col2_iter2:
                st.markdown("""#### Pilotes podium VS prédictions""")
                st._legacy_dataframe(podium_real_dt.style.apply(df_background_color, axis=1), height=735)
    

    st.write('---')

    st.markdown(
        """
        ## Top 3 + Podium

        Après avoir exploré l’approche « podium », nous avons pensé à 2 possibilités pour obtenir un classement top 3 :

        <b><u>Option 1</u> :</b><br>
        A partir de la méthodologie « podium », on récupère les 3 valeurs maximales des probabilités de la classe 1 et on associe les valeur 1/2/3 dans les prédictions.


        <b><u>Option 2</u> :</b><br>
        On combine les probabilités obtenues pour le top 3 et le podium de la manière suivante :
        - proba_1 = proba_1_podium + proba_1_top3
        - proba_2 = proba_1_podium + proba_1_top3 + proba_2_top3
        - proba_3 = proba_1_podium + proba_1_top3 + proba_2_top3 + proba_3_top3

        Ensuite, on détermine les positions 1/2/3 comme suit :
        - La 1ere place correspondant à la valeur max de proba_1
        - On filtre la ligne de la 1ere place qui a été attribuée et on récupère la valeur max de proba_2 pour la 2e place.
        - On filtre la ligne de la 2e place attribuée et on récupère la valeur max de proba_3 pour déterminer la 3e place.

        """, unsafe_allow_html=True)
    
    model_selector_3 = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision'), key='iter3',
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)

    if model_selector_3 == 'Régression logistique':
        st.write('reg log')
    

    elif model_selector_3 == 'Forêt aléatoire':
        st.write('random forest')
    

    elif model_selector_3 == 'Arbre de décision':
        st.write('decision tree')
    

    # ----------------------------
    # Conclusion
    # ----------------------------
    st.write('---')
    st.markdown(
        """
        ## Récap

        """)