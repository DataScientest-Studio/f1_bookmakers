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
from sklearn.metrics import classification_report

title = "Modélisations"
sidebar_name = "Modélisation - Top 3"

def run():

    #st.markdown('<style>section[tabindex="0"] div[data-testid="stHorizontalBlock"]:last-child div[data-testid="column"].css-keje6w {border-right: 4px solid var(--gray-color); border-top: 4px solid var(--gray-color); border-top-right-radius: 15px; padding: 5px 5px 0 0;}</style>', unsafe_allow_html=True)

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
            log_reg = LogisticRegression(C=C_param_selector, random_state=1430)
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
            
            
            # dataframe des pilotes réels
            podium_real_log = df_test_proba[df_test_proba["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_log = podium_real_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
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
            
            score_top_3_ranked_log = np.around((podium_real_log[podium_real_log['Top 3 ranked']=='✅'].shape[0] / podium_real_log.shape[0]) * 100, 2)
            score_top_3_unranked_log = np.around((podium_real_log[podium_real_log['Top 3 unranked']=='✔'].shape[0] / podium_real_log.shape[0]) * 100, 2)


            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_df)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_log))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_log))

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
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param3-iter1')

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


            # dataframe des pilotes réels
            podium_real_rf = df_test_prob_rf[df_test_prob_rf["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_rf = podium_real_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
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
            
            score_top_3_ranked_rf = np.around((podium_real_rf[podium_real_rf['Top 3 ranked']=='✅'].shape[0] / podium_real_rf.shape[0]) * 100, 2)
            score_top_3_unranked_rf = np.around((podium_real_rf[podium_real_rf['Top 3 unranked']=='✔'].shape[0] / podium_real_rf.shape[0]) * 100, 2)

            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_rf)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_rf_df)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_rf))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_rf))

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


            # dataframe des pilotes réels
            podium_real_dt = df_test_prob_dt[df_test_prob_dt["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_dt = podium_real_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
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
            
            score_top_3_ranked_dt = np.around((podium_real_dt[podium_real_dt['Top 3 ranked']=='✅'].shape[0] / podium_real_dt.shape[0]) * 100, 2)
            score_top_3_unranked_dt = np.around((podium_real_dt[podium_real_dt['Top 3 unranked']=='✔'].shape[0] / podium_real_dt.shape[0]) * 100, 2)

            col1_iter1, col2_iter1 = st.columns(2)
            with col1_iter1:
                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_dt)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_dt_df)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_dt))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_dt))

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

            # dataframe des pilotes réels
            podium_real_log = df_test_proba[df_test_proba["podium"]==1][['round', 'driverId']]
            podium_real_log = podium_real_log.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .sort_values(by=['round'])\
                                                        .drop(['driverId'], axis=1)\
                                                        .reset_index(drop=True)

            # dataframe des pilotes prédits
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
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param3-iter2')

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
            

            # dataframe des pilotes réels
            podium_real_rf = df_test_prob_rf[df_test_prob_rf["podium"]==1][['round', 'driverId']]
            podium_real_rf = podium_real_rf.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .sort_values(by=['round'])\
                                                        .drop(['driverId'], axis=1)\
                                                        .reset_index(drop=True)

            # dataframe des pilotes prédits
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
            

            # dataframe des pilotes réels
            podium_real_dt = df_test_prob_dt[df_test_prob_dt["podium"]==1][['round', 'driverId']]
            podium_real_dt = podium_real_dt.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .sort_values(by=['round'])\
                                                        .drop(['driverId'], axis=1)\
                                                        .reset_index(drop=True)

            # dataframe des pilotes prédits
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
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter3, param_col2_iter3, param_col3_iter3, param_col4_iter3 = st.columns(4)

        with param_col1_iter3:
            st.write('Modèle Podium')
            C_param_selector_opt1 = st.selectbox(label='C', options=(0.0001, 0.001, 0.01, 0.1, 1, 10), index=0, key='log-iter3-opt1')
        
        with param_col3_iter3:
            st.write('Modèle Top 3')
            C_param_selector_opt2 = st.selectbox(label='C', options=(0.0001, 0.001, 0.01, 0.1, 1, 10), index=1, key='log-iter3-opt2')

        if st.button('Résultats', key='log-iter3'):  

            st.write('---')

            # ---------
            # Option 1
            # ---------
            # instanciation modèle
            log_reg_podium = LogisticRegression(C=C_param_selector_opt1)
            log_reg_podium.fit(X_ro_podium, y_ro_podium)

            # probabilité avec predict_proba
            y_pred_log_ro_podium = log_reg_podium.predict_proba(X_test_scaled)
            df_y_pred_log_ro_podium = pd.DataFrame(y_pred_log_ro_podium, columns=['proba_0', 'proba_1'])

            # création dataframe des résultats
            df_test_prob_log_ro_podium = pd.concat([df_test.reset_index(), df_y_pred_log_ro_podium], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_log_ro_podium['prediction'] = 0

            # liste des courses par raceId
            raceId_list_opt1 = df_test_prob_log_ro_podium['raceId'].unique()
            
            # boucle sur chaque course
            for i in raceId_list_opt1:
                df_temp = df_test_prob_log_ro_podium[df_test_prob_log_ro_podium['raceId']==i]   # filtre les données de la course
                index_podium = df_temp.loc[:,'proba_1'].sort_values(ascending=False)[:3].index  # on récupère l'index des 3 valeurs max
                df_test_prob_log_ro_podium.loc[index_podium[0], 'prediction'] = 1               # on affecte valeur 1 dans prediction pour la 1ère valeur max
                df_test_prob_log_ro_podium.loc[index_podium[1], 'prediction'] = 2               # on affecte valeur 2 dans prediction pour la 2ère valeur max
                df_test_prob_log_ro_podium.loc[index_podium[2], 'prediction'] = 3               # on affecte valeur 3 dans prediction pour la 3ère valeur max


            # rapport classification et matrice de confusion
            confusion_matrix_3_opt1 = pd.crosstab(df_test_prob_log_ro_podium['positionOrder'], df_test_prob_log_ro_podium['prediction'])
            confusion_matrix_3_opt1.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_3_opt1.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_df_3_opt1 = pd.DataFrame(classification_report(df_test_prob_log_ro_podium['positionOrder'], df_test_prob_log_ro_podium['prediction'], output_dict=True)).T[:4]
            classif_report_df_3_opt1['support'] = classif_report_df_3_opt1['support'].astype('int')
            classif_report_df_3_opt1.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

            # dataframe des vainqueurs réels
            podium_real_log_opt1 = df_test_prob_log_ro_podium[df_test_prob_log_ro_podium["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_log_opt1 = podium_real_log_opt1.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des vainqueurs prédits
            podium_predicted_log_opt1 = df_test_prob_log_ro_podium[df_test_prob_log_ro_podium["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_log_opt1 = podium_predicted_log_opt1.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_log_opt1 = podium_real_log_opt1.merge(right=podium_predicted_log_opt1, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_log_opt1['Top 3 ranked'] = podium_real_log_opt1.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list_opt1 = podium_real_log_opt1['round'].unique()
            podium_real_log_opt1['Top 3 unranked'] = '❌'

            for i in rounds_list_opt1:
                temp_df = podium_real_log_opt1[podium_real_log_opt1['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_log_opt1.loc[index_driver, 'Top 3 unranked'] = '✔'
            
            score_top_3_ranked_log_opt1 = np.around((podium_real_log_opt1[podium_real_log_opt1['Top 3 ranked']=='✅'].shape[0] / podium_real_log_opt1.shape[0]) * 100, 2)
            score_top_3_unranked_log_opt1 = np.around((podium_real_log_opt1[podium_real_log_opt1['Top 3 unranked']=='✔'].shape[0] / podium_real_log_opt1.shape[0]) * 100, 2)
            

            # ---------
            # Option 2
            # ---------
            # instanciation modèle
            log_reg_top3 = LogisticRegression(C=C_param_selector_opt2)
            log_reg_top3.fit(X_ro_top3, y_ro_top3)

            # probabilité avec predict_proba
            y_pred_log_ro_top3 = log_reg_top3.predict_proba(X_test_scaled)
            df_y_pred_log_ro_top3 = pd.DataFrame(y_pred_log_ro_top3, columns=['proba_0_top3', 'proba_1_top3', 'proba_2_top3', 'proba_3_top3'])

            y_pred_log_ro_podium = log_reg_podium.predict_proba(X_test_scaled)
            df_y_pred_log_ro_podium = pd.DataFrame(y_pred_log_ro_podium, columns=['proba_0_podium', 'proba_1_podium'])

            # création dataframe des résultats avec fusion probabilités top 3 et podium
            df_test_prob_log_ro_podium_top3 = pd.concat([df_test.reset_index(), df_y_pred_log_ro_podium, df_y_pred_log_ro_top3], axis=1)

            # On combine les probabilités obtenues pour le top 3 et le podium :
            #    proba_1 = proba_1_podium + proba_1_top3
            #    proba_2 = proba_1_podium + proba_1_top3 + proba_2_top3
            #    proba_3 = proba_1_podium + proba_1_top3 + proba_2_top3 + proba_3_top3
            df_test_prob_log_ro_podium_top3['proba_1'] = df_test_prob_log_ro_podium_top3['proba_1_podium'] + df_test_prob_log_ro_podium_top3['proba_1_top3']
            df_test_prob_log_ro_podium_top3['proba_2'] = df_test_prob_log_ro_podium_top3['proba_1'] + df_test_prob_log_ro_podium_top3['proba_2_top3']
            df_test_prob_log_ro_podium_top3['proba_3'] = df_test_prob_log_ro_podium_top3['proba_2'] + df_test_prob_log_ro_podium_top3['proba_3_top3']

            # ajout colonne prediction initialisée à 0
            df_test_prob_log_ro_podium_top3['prediction'] = 0

            # liste des courses par raceId
            raceId_list_opt2 = df_test_prob_log_ro_podium_top3['raceId'].unique()
            
            # boucle sur chaque course
            for i in raceId_list_opt2:
                df_temp = df_test_prob_log_ro_podium_top3[df_test_prob_log_ro_podium_top3['raceId']==i]  # filtre les données de la course
                max_proba_1 = df_temp['proba_1'].max()                                                   # on identifie la valeur max de la colonne proba_1
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index                       # on récupère l'index de la valeur max
                df_test_prob_log_ro_podium_top3.loc[index_max_proba_1, 'prediction'] = 1                 # on affecte valeur 1 dans prediction à l'index max
                
                df_temp = df_test_prob_log_ro_podium_top3[df_test_prob_log_ro_podium_top3['raceId']==i]  # on récupère les données de la course mises à jour
                max_proba_2 = df_temp[df_temp['prediction']==0]['proba_2'].max()                         # on identifie la valeur max de la colonne proba_2 en filtrant la ligne affectée à 1 précédemment
                index_max_proba_2 = df_temp[df_temp['proba_2']==max_proba_2].index                       # on récupère l'index de la valeur max
                df_test_prob_log_ro_podium_top3.loc[index_max_proba_2, 'prediction'] = 2                 # on affecte valeur 2 dans prediction à l'index max
                
                df_temp = df_test_prob_log_ro_podium_top3[df_test_prob_log_ro_podium_top3['raceId']==i]  # on récupère les données de la course mises à jour
                max_proba_3 = df_temp[df_temp['prediction']==0]['proba_3'].max()                         # on identifie la valeur max de la colonne proba_3 en filtrant les lignes affectées à 1 et 2 précédemment
                index_max_proba_3 = df_temp[df_temp['proba_3']==max_proba_3].index                       # on récupère l'index de la valeur max
                df_test_prob_log_ro_podium_top3.loc[index_max_proba_3, 'prediction'] = 3                 # on affecte valeur 3 dans prediction à l'index max


            # rapport classification et matrice de confusion
            confusion_matrix_3_opt2 = pd.crosstab(df_test_prob_log_ro_podium_top3['positionOrder'], df_test_prob_log_ro_podium_top3['prediction'])
            confusion_matrix_3_opt2.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_3_opt2.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_df_3_opt2 = pd.DataFrame(classification_report(df_test_prob_log_ro_podium_top3['positionOrder'], df_test_prob_log_ro_podium_top3['prediction'], output_dict=True)).T[:4]
            classif_report_df_3_opt2['support'] = classif_report_df_3_opt2['support'].astype('int')
            classif_report_df_3_opt2.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

            # dataframe des pilotes réels
            podium_real_log_opt2 = df_test_prob_log_ro_podium_top3[df_test_prob_log_ro_podium_top3["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_log_opt2 = podium_real_log_opt2.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
            podium_predicted_log_opt2 = df_test_prob_log_ro_podium_top3[df_test_prob_log_ro_podium_top3["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_log_opt2 = podium_predicted_log_opt2.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_log_opt2 = podium_real_log_opt2.merge(right=podium_predicted_log_opt2, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_log_opt2['Top 3 ranked'] = podium_real_log_opt2.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list_opt1 = podium_real_log_opt2['round'].unique()
            podium_real_log_opt2['Top 3 unranked'] = '❌'

            for i in rounds_list_opt1:
                temp_df = podium_real_log_opt2[podium_real_log_opt2['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_log_opt2.loc[index_driver, 'Top 3 unranked'] = '✔'
            
            
            score_top_3_ranked_log_opt2 = np.around((podium_real_log_opt2[podium_real_log_opt2['Top 3 ranked']=='✅'].shape[0] / podium_real_log_opt2.shape[0]) * 100, 2)
            score_top_3_unranked_log_opt2 = np.around((podium_real_log_opt2[podium_real_log_opt2['Top 3 unranked']=='✔'].shape[0] / podium_real_log_opt2.shape[0]) * 100, 2)

            # ----------
            # Résultats
            # ----------
            col1_iter3, col2_iter3 = st.columns(2)
            with col1_iter3:
                st.markdown("""### Option 1""")

                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_3_opt1)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_df_3_opt1)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_log_opt1))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_log_opt1))

                st.write('---')

                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_log_opt1.style.apply(df_background_color, axis=1), height=735)

            with col2_iter3:
                st.markdown("""### Option 2""")

                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_3_opt2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_df_3_opt2)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_log_opt2))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_log_opt2))

                st.write('---')

                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_log_opt2.style.apply(df_background_color, axis=1), height=735)
    

    elif model_selector_3 == 'Forêt aléatoire':
        # ----------------------
        # Modèle Forêt aléatoire
        # ----------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter3, param_col2_iter3, param_col3_iter3, param_col4_iter3 = st.columns(4)

        with param_col1_iter3:
            st.write('Modèle Podium')
            n_estimators_param_selector_opt1 = st.selectbox(label='n_estimators', options=(10, 50, 100, 250), index=2, key='rf_param1-iter3-opt1')
            min_samples_leaf_param_selector_opt1 = st.selectbox(label='min_samples_leaf', options=(1, 3, 5), index=0, key='rf_param2-iter3-opt1')
            max_features_param_selector_opt1 = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param3-iter3-opt1')
        
        with param_col3_iter3:
            st.write('Modèle Top 3')
            n_estimators_param_selector_opt2 = st.selectbox(label='n_estimators', options=(10, 50, 100, 250), index=2, key='rf_param1-iter3-opt2')
            min_samples_leaf_param_selector_opt2 = st.selectbox(label='min_samples_leaf', options=(1, 3, 5), index=0, key='rf_param2-iter3-opt2')
            max_features_param_selector_opt2 = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param3-iter3-opt2')

        if st.button('Résultats', key='rf-iter3'):

            st.write('---')
        
            # ---------
            # Option 1
            # ---------
            # instanciation modèle
            rf_podium = RandomForestClassifier(n_jobs=-1, max_features = max_features_param_selector_opt1, min_samples_leaf = min_samples_leaf_param_selector_opt1,
                                            n_estimators = n_estimators_param_selector_opt1, random_state=1430)
            rf_podium.fit(X_train_scaled, y_train_podium)

            # probabilité avec predict_proba
            y_pred_rf_podium = rf_podium.predict_proba(X_test_scaled)
            df_y_pred_rf_podium = pd.DataFrame(y_pred_rf_podium, columns=['proba_0', 'proba_1'])

            # création dataframe des résultats
            df_test_prob_rf_podium = pd.concat([df_test.reset_index(), df_y_pred_rf_podium], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_rf_podium['prediction'] = 0

            # liste des courses par raceId
            raceId_listRF_opt1 = df_test_prob_rf_podium['raceId'].unique()

            for i in raceId_listRF_opt1:
                df_temp = df_test_prob_rf_podium[df_test_prob_rf_podium['raceId']==i]                         # filtre les données de la course
                index_podium = df_temp.loc[:,'proba_1'].sort_values(ascending=False)[:3].index  # on récupère l'index des 3 valeurs max
                df_test_prob_rf_podium.loc[index_podium[0], 'prediction'] = 1                          # on affecte valeur 1 dans prediction pour la 1ère valeur max
                df_test_prob_rf_podium.loc[index_podium[1], 'prediction'] = 2                          # on affecte valeur 2 dans prediction pour la 2ère valeur max
                df_test_prob_rf_podium.loc[index_podium[2], 'prediction'] = 3                          # on affecte valeur 3 dans prediction pour la 3ère valeur max

            # rapport classification et matrice de confusion
            confusion_matrix_rf_3_opt1 = pd.crosstab(df_test_prob_rf_podium['positionOrder'], df_test_prob_rf_podium['prediction'])
            confusion_matrix_rf_3_opt1.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_rf_3_opt1.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_rf_df_3_opt1 = pd.DataFrame(classification_report(df_test_prob_rf_podium['positionOrder'], df_test_prob_rf_podium['prediction'], output_dict=True)).T[:4]
            classif_report_rf_df_3_opt1['support'] = classif_report_rf_df_3_opt1['support'].astype('int')
            classif_report_rf_df_3_opt1.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']


            # dataframe des pilotes réels
            podium_real_rf_opt1 = df_test_prob_rf_podium[df_test_prob_rf_podium["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_rf_opt1 = podium_real_rf_opt1.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
            podium_predicted_rf_opt1 = df_test_prob_rf_podium[df_test_prob_rf_podium["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_rf_opt1 = podium_predicted_rf_opt1.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_rf_opt1 = podium_real_rf_opt1.merge(right=podium_predicted_rf_opt1, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_rf_opt1['Top 3 ranked'] = podium_real_rf_opt1.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list_opt1 = podium_real_rf_opt1['round'].unique()
            podium_real_rf_opt1['Top 3 unranked'] = '❌'

            for i in rounds_list_opt1:
                temp_df = podium_real_rf_opt1[podium_real_rf_opt1['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_rf_opt1.loc[index_driver, 'Top 3 unranked'] = '✔'
            
            score_top_3_ranked_rf_opt1 = np.around((podium_real_rf_opt1[podium_real_rf_opt1['Top 3 ranked']=='✅'].shape[0] / podium_real_rf_opt1.shape[0]) * 100, 2)
            score_top_3_unranked_rf_opt1 = np.around((podium_real_rf_opt1[podium_real_rf_opt1['Top 3 unranked']=='✔'].shape[0] / podium_real_rf_opt1.shape[0]) * 100, 2)
            
            # ---------
            # Option 2
            # ---------
            # instanciation modèle
            rf_top3 = RandomForestClassifier(n_jobs=-1, max_features = max_features_param_selector_opt2, min_samples_leaf = min_samples_leaf_param_selector_opt2,
                                            n_estimators = n_estimators_param_selector_opt2, random_state=1430)
            rf_top3.fit(X_train_scaled, y_train_top3)

            # probabilité avec predict_proba
            y_pred_rf_top3 = rf_top3.predict_proba(X_test_scaled)
            df_y_pred_rf_top3 = pd.DataFrame(y_pred_rf_top3, columns=['proba_0_top3', 'proba_1_top3', 'proba_2_top3', 'proba_3_top3'])

            y_pred_rf_podium = rf_podium.predict_proba(X_test_scaled)
            df_y_pred_rf_podium = pd.DataFrame(y_pred_rf_podium, columns=['proba_0_podium', 'proba_1_podium'])

            # création dataframe des résultats avec fusion probabilités top 3 et podium
            df_test_prob_rf_podium_top3 = pd.concat([df_test.reset_index(), df_y_pred_rf_podium, df_y_pred_rf_top3], axis=1)

            # On combine les probabilités obtenues pour le top 3 et le podium :
            #    proba_1 = proba_1_podium + proba_1_top3
            #    proba_2 = proba_1_podium + proba_1_top3 + proba_2_top3
            #    proba_3 = proba_1_podium + proba_1_top3 + proba_2_top3 + proba_3_top3
            df_test_prob_rf_podium_top3['proba_1'] = df_test_prob_rf_podium_top3['proba_1_podium'] + df_test_prob_rf_podium_top3['proba_1_top3']
            df_test_prob_rf_podium_top3['proba_2'] = df_test_prob_rf_podium_top3['proba_1'] + df_test_prob_rf_podium_top3['proba_2_top3']
            df_test_prob_rf_podium_top3['proba_3'] = df_test_prob_rf_podium_top3['proba_2'] + df_test_prob_rf_podium_top3['proba_3_top3']

            # ajout colonne prediction initialisée à 0
            df_test_prob_rf_podium_top3['prediction'] = 0

           # liste des courses par raceId
            raceId_listRF_opt2 = df_test_prob_rf_podium_top3['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_listRF_opt2:
                df_temp = df_test_prob_rf_podium_top3[df_test_prob_rf_podium_top3['raceId']==i]  # filtre les données de la course
                max_proba_1 = df_temp['proba_1'].max()                                           # on identifie la valeur max de la colonne proba_1
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index               # on récupère l'index de la valeur max
                df_test_prob_rf_podium_top3.loc[index_max_proba_1, 'prediction'] = 1             # on affecte valeur 1 dans prediction à l'index max
                
                df_temp = df_test_prob_rf_podium_top3[df_test_prob_rf_podium_top3['raceId']==i]  # on récupère les données de la course mises à jour
                max_proba_2 = df_temp[df_temp['prediction']==0]['proba_2'].max()                 # on identifie la valeur max de la colonne proba_2 en filtrant la ligne affectée à 1 précédemment
                index_max_proba_2 = df_temp[df_temp['proba_2']==max_proba_2].index               # on récupère l'index de la valeur max
                df_test_prob_rf_podium_top3.loc[index_max_proba_2, 'prediction'] = 2             # on affecte valeur 2 dans prediction à l'index max
                
                df_temp = df_test_prob_rf_podium_top3[df_test_prob_rf_podium_top3['raceId']==i]  # on récupère les données de la course mises à jour
                max_proba_3 = df_temp[df_temp['prediction']==0]['proba_3'].max()                 # on identifie la valeur max de la colonne proba_3 en filtrant les lignes affectées à 1 et 2 précédemment
                index_max_proba_3 = df_temp[df_temp['proba_3']==max_proba_3].index               # on récupère l'index de la valeur max
                df_test_prob_rf_podium_top3.loc[index_max_proba_3, 'prediction'] = 3             # on affecte valeur 3 dans prediction à l'index max
            
            # rapport classification et matrice de confusion
            confusion_matrix_rf_3_opt2 = pd.crosstab(df_test_prob_rf_podium_top3['positionOrder'], df_test_prob_rf_podium_top3['prediction'])
            confusion_matrix_rf_3_opt2.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_rf_3_opt2.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_rf_df_3_opt2 = pd.DataFrame(classification_report(df_test_prob_rf_podium_top3['positionOrder'], df_test_prob_rf_podium_top3['prediction'], output_dict=True)).T[:4]
            classif_report_rf_df_3_opt2['support'] = classif_report_rf_df_3_opt2['support'].astype('int')
            classif_report_rf_df_3_opt2.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

            # dataframe des pilotes réels
            podium_real_rf_opt2 = df_test_prob_rf_podium_top3[df_test_prob_rf_podium_top3["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_rf_opt2 = podium_real_rf_opt2.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
            podium_predicted_rf_opt2 = df_test_prob_rf_podium_top3[df_test_prob_rf_podium_top3["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_rf_opt2 = podium_predicted_rf_opt2.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_rf_opt2 = podium_real_rf_opt2.merge(right=podium_predicted_rf_opt2, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_rf_opt2['Top 3 ranked'] = podium_real_rf_opt2.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list_opt2 = podium_real_rf_opt2['round'].unique()
            podium_real_rf_opt2['Top 3 unranked'] = '❌'

            for i in rounds_list_opt2:
                temp_df = podium_real_rf_opt2[podium_real_rf_opt2['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_rf_opt2.loc[index_driver, 'Top 3 unranked'] = '✔'
            
            score_top_3_ranked_rf_opt2 = np.around((podium_real_rf_opt2[podium_real_rf_opt2['Top 3 ranked']=='✅'].shape[0] / podium_real_rf_opt2.shape[0]) * 100, 2)
            score_top_3_unranked_rf_opt2 = np.around((podium_real_rf_opt2[podium_real_rf_opt2['Top 3 unranked']=='✔'].shape[0] / podium_real_rf_opt2.shape[0]) * 100, 2)
            

            # ----------
            # Résultats
            # ----------
            col1_iter3, col2_iter3 = st.columns(2)
            with col1_iter3:
                st.markdown("""### Option 1""")

                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_rf_3_opt1)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_rf_df_3_opt1)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_rf_opt1))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_rf_opt1))

                st.write('---')

                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_rf_opt1.style.apply(df_background_color, axis=1), height=735)
            
            with col2_iter3:
                st.markdown("""### Option 2""")

                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_rf_3_opt2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_rf_df_3_opt2)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_rf_opt2))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_rf_opt2))

                st.write('---')

                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_rf_opt2.style.apply(df_background_color, axis=1), height=735)

    
    elif model_selector_3 == 'Arbre de décision':
        # ------------------------
        # Modèle Arbre de décision
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1_iter3, param_col2_iter3, param_col3_iter3, param_col4_iter3 = st.columns(4)

        with param_col1_iter3:
            st.write('Modèle Podium')
            criterion_param_selector_opt1 = st.selectbox(label='criterion', options=('entropy', 'gini'), index=0, key='dt_param1-iter3-opt1')
            max_depth_param_selector_opt1 = st.selectbox(label='max_depth', options=(1, 2, 3, 4, 5, 6, 7), index=5, key='dt_param2-iter3-opt1')
            
        with param_col3_iter3:
            st.write('Modèle Top 3')
            criterion_param_selector_opt1 = st.selectbox(label='criterion', options=('entropy', 'gini'), index=0, key='dt_param1-iter3-opt2')
            max_depth_param_selector_opt1 = st.selectbox(label='max_depth', options=(1, 2, 3, 4, 5, 6, 7), index=4, key='dt_param2-iter3-opt2')

        if st.button('Résultats', key='dt-iter3'):

            st.write('---')

            # ---------
            # Option 1
            # ---------
            # instanciation modèle
            dt_podium = DecisionTreeClassifier(criterion=criterion_param_selector_opt1, max_depth=max_depth_param_selector_opt1, random_state=143)
            dt_podium.fit(X_ro_podium, y_ro_podium)

            # probabilité avec predict_proba
            y_pred_dt_podium = dt_podium.predict_proba(X_test_scaled)
            df_y_pred_dt_podium = pd.DataFrame(y_pred_dt_podium, columns=['proba_0', 'proba_1'])

            # création dataframe des résultats
            df_test_prob_dt_podium = pd.concat([df_test.reset_index(), df_y_pred_dt_podium], axis=1)
            # ajout colonne prediction initialisée à 0
            df_test_prob_dt_podium['prediction'] = 0

            # liste des courses par raceId
            raceId_list_dt_opt1 = df_test_prob_dt_podium['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list_dt_opt1:
                df_temp = df_test_prob_dt_podium[df_test_prob_dt_podium['raceId']==i]                         # filtre les données de la course
                index_podium = df_temp.loc[:,'proba_1'].sort_values(ascending=False)[:3].index  # on récupère l'index des 3 valeurs max
                df_test_prob_dt_podium.loc[index_podium[0], 'prediction'] = 1                          # on affecte valeur 1 dans prediction pour la 1ère valeur max
                df_test_prob_dt_podium.loc[index_podium[1], 'prediction'] = 2                          # on affecte valeur 2 dans prediction pour la 2ère valeur max
                df_test_prob_dt_podium.loc[index_podium[2], 'prediction'] = 3                          # on affecte valeur 3 dans prediction pour la 3ère valeur max
            
            # rapport classification et matrice de confusion
            confusion_matrix_dt_3_opt1 = pd.crosstab(df_test_prob_dt_podium['positionOrder'], df_test_prob_dt_podium['prediction'])
            confusion_matrix_dt_3_opt1.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_dt_3_opt1.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_dt_df_3_opt1 = pd.DataFrame(classification_report(df_test_prob_dt_podium['positionOrder'], df_test_prob_dt_podium['prediction'], output_dict=True)).T[:4]
            classif_report_dt_df_3_opt1['support'] = classif_report_dt_df_3_opt1['support'].astype('int')
            classif_report_dt_df_3_opt1.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

            # dataframe des pilotes réels
            podium_real_dt_opt1 = df_test_prob_dt_podium[df_test_prob_dt_podium["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_dt_opt1 = podium_real_dt_opt1.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
            podium_predicted_dt_opt1 = df_test_prob_dt_podium[df_test_prob_dt_podium["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_dt_opt1 = podium_predicted_dt_opt1.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_dt_opt1 = podium_real_dt_opt1.merge(right=podium_predicted_dt_opt1, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_dt_opt1['Top 3 ranked'] = podium_real_dt_opt1.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list_opt1 = podium_real_dt_opt1['round'].unique()
            podium_real_dt_opt1['Top 3 unranked'] = '❌'

            for i in rounds_list_opt1:
                temp_df = podium_real_dt_opt1[podium_real_dt_opt1['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_dt_opt1.loc[index_driver, 'Top 3 unranked'] = '✔'
            
            score_top_3_ranked_dt_opt1 = np.around((podium_real_dt_opt1[podium_real_dt_opt1['Top 3 ranked']=='✅'].shape[0] / podium_real_dt_opt1.shape[0]) * 100, 2)
            score_top_3_unranked_dt_opt1 = np.around((podium_real_dt_opt1[podium_real_dt_opt1['Top 3 unranked']=='✔'].shape[0] / podium_real_dt_opt1.shape[0]) * 100, 2)
            
            # ---------
            # Option 2
            # ---------
            # instanciation modèle
            dt_top3 = DecisionTreeClassifier(criterion=criterion_param_selector_opt1, max_depth=max_depth_param_selector_opt1, random_state=143)
            dt_top3.fit(X_ro_top3, y_ro_top3)

            # probabilité avec predict_proba
            y_pred_dt_top3 = dt_top3.predict_proba(X_test_scaled)
            df_y_pred_dt_top3 = pd.DataFrame(y_pred_dt_top3, columns=['proba_0_top3', 'proba_1_top3', 'proba_2_top3', 'proba_3_top3'])

            y_pred_dt_podium = dt_podium.predict_proba(X_test_scaled)
            df_y_pred_dt_podium = pd.DataFrame(y_pred_dt_podium, columns=['proba_0_podium', 'proba_1_podium'])

            # création dataframe des résultats avec fusion probabilités top 3 et podium
            df_test_prob_dt_podium_top3 = pd.concat([df_test.reset_index(), df_y_pred_dt_podium, df_y_pred_dt_top3], axis=1)

            # On combine les probabilités obtenues pour le top 3 et le podium :
            #    proba_1 = proba_1_podium + proba_1_top3
            #    proba_2 = proba_1_podium + proba_1_top3 + proba_2_top3
            #    proba_3 = proba_1_podium + proba_1_top3 + proba_2_top3 + proba_3_top3
            df_test_prob_dt_podium_top3['proba_1'] = df_test_prob_dt_podium_top3['proba_1_podium'] + df_test_prob_dt_podium_top3['proba_1_top3']
            df_test_prob_dt_podium_top3['proba_2'] = df_test_prob_dt_podium_top3['proba_1'] + df_test_prob_dt_podium_top3['proba_2_top3']
            df_test_prob_dt_podium_top3['proba_3'] = df_test_prob_dt_podium_top3['proba_2'] + df_test_prob_dt_podium_top3['proba_3_top3']

            # ajout colonne prediction initialisée à 0
            df_test_prob_dt_podium_top3['prediction'] = 0

            # liste des courses par raceId
            raceId_list_dt_opt2 = df_test_prob_dt_podium_top3['raceId'].unique()

            # boucle sur chaque course
            for i in raceId_list_dt_opt2:
                df_temp = df_test_prob_dt_podium_top3[df_test_prob_dt_podium_top3['raceId']==i]  # filtre les données de la course
                max_proba_1 = df_temp['proba_1'].max()                                           # on identifie la valeur max de la colonne proba_1
                index_max_proba_1 = df_temp[df_temp['proba_1']==max_proba_1].index               # on récupère l'index de la valeur max
                df_test_prob_dt_podium_top3.loc[index_max_proba_1, 'prediction'] = 1             # on affecte valeur 1 dans prediction à l'index max
                
                df_temp = df_test_prob_dt_podium_top3[df_test_prob_dt_podium_top3['raceId']==i]  # on récupère les données de la course mises à jour
                max_proba_2 = df_temp[df_temp['prediction']==0]['proba_2'].max()                 # on identifie la valeur max de la colonne proba_2 en filtrant la ligne affectée à 1 précédemment
                index_max_proba_2 = df_temp[df_temp['proba_2']==max_proba_2].index               # on récupère l'index de la valeur max
                df_test_prob_dt_podium_top3.loc[index_max_proba_2, 'prediction'] = 2             # on affecte valeur 2 dans prediction à l'index max
                
                df_temp = df_test_prob_dt_podium_top3[df_test_prob_dt_podium_top3['raceId']==i]  # on récupère les données de la course mises à jour
                max_proba_3 = df_temp[df_temp['prediction']==0]['proba_3'].max()                 # on identifie la valeur max de la colonne proba_3 en filtrant les lignes affectées à 1 et 2 précédemment
                index_max_proba_3 = df_temp[df_temp['proba_3']==max_proba_3].index               # on récupère l'index de la valeur max
                df_test_prob_dt_podium_top3.loc[index_max_proba_3, 'prediction'] = 3             # on affecte valeur 3 dans prediction à l'index max
            
            # rapport classification et matrice de confusion
            confusion_matrix_dt_3_opt2 = pd.crosstab(df_test_prob_dt_podium_top3['positionOrder'], df_test_prob_dt_podium_top3['prediction'])
            confusion_matrix_dt_3_opt2.columns = ['Pred. 0', 'Pred. 1', 'Pred. 2', 'Pred. 3']
            confusion_matrix_dt_3_opt2.index = ['Real val. 0', 'Real val. 1', 'Real val. 2', 'Real val. 3']

            classif_report_dt_df_3_opt2 = pd.DataFrame(classification_report(df_test_prob_dt_podium_top3['positionOrder'], df_test_prob_dt_podium_top3['prediction'], output_dict=True)).T[:4]
            classif_report_dt_df_3_opt2['support'] = classif_report_dt_df_3_opt2['support'].astype('int')
            classif_report_dt_df_3_opt2.index = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

            # dataframe des pilotes réels
            podium_real_dt_opt2 = df_test_prob_dt_podium_top3[df_test_prob_dt_podium_top3["positionOrder"]!=0][['round', 'positionOrder', 'driverId']]
            podium_real_dt_opt2 = podium_real_dt_opt2.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                        .rename(columns={'surname' : 'Driver'})\
                                                        .drop(['driverId'], axis=1)

            # dataframe des pilotes prédits
            podium_predicted_dt_opt2 = df_test_prob_dt_podium_top3[df_test_prob_dt_podium_top3["prediction"]!=0][['round', 'prediction', 'driverId']]
            podium_predicted_dt_opt2 = podium_predicted_dt_opt2.merge(right=drivers_data[['driverId', 'surname']], on='driverId')\
                                                            .rename(columns={'surname' : 'Predicted driver'})\
                                                            .drop(['driverId'], axis=1)

            # fusion des 2 dataframes
            podium_real_dt_opt2 = podium_real_dt_opt2.merge(right=podium_predicted_dt_opt2, left_on=['round', 'positionOrder'], right_on=['round', 'prediction'])\
                                .sort_values(by=['round', 'positionOrder'])\
                                .rename(columns={'positionOrder' : 'position'})\
                                .drop(['prediction'], axis=1)\
                                .reset_index(drop=True)
            
            podium_real_dt_opt2['Top 3 ranked'] = podium_real_dt_opt2.apply(lambda row: '✅' if row['Driver']==row['Predicted driver'] else '❌', axis=1)

            rounds_list_opt2 = podium_real_dt_opt2['round'].unique()
            podium_real_dt_opt2['Top 3 unranked'] = '❌'

            for i in rounds_list_opt2:
                temp_df = podium_real_dt_opt2[podium_real_dt_opt2['round']==i]
                drivers_list = list(temp_df['Driver'])
                for driver in temp_df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = temp_df[temp_df['Predicted driver']==driver].index
                        podium_real_dt_opt2.loc[index_driver, 'Top 3 unranked'] = '✔'
            
            score_top_3_ranked_dt_opt2 = np.around((podium_real_dt_opt2[podium_real_dt_opt2['Top 3 ranked']=='✅'].shape[0] / podium_real_dt_opt2.shape[0]) * 100, 2)
            score_top_3_unranked_dt_opt2 = np.around((podium_real_dt_opt2[podium_real_dt_opt2['Top 3 unranked']=='✔'].shape[0] / podium_real_dt_opt2.shape[0]) * 100, 2)

            
            # ----------
            # Résultats
            # ----------
            col1_iter3, col2_iter3 = st.columns(2)
            with col1_iter3:
                st.markdown("""### Option 1""")

                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_dt_3_opt1)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_dt_df_3_opt1)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_dt_opt1))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_dt_opt1))

                st.write('---')

                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_dt_opt1.style.apply(df_background_color, axis=1), height=735)
            
            with col2_iter3:
                st.markdown("""### Option 2""")

                st.markdown("""#### Matrice de confusion""")
                st.dataframe(confusion_matrix_dt_3_opt2)

                st.write('---')

                st.markdown("""#### Rapport de classification""")
                st.write(classif_report_dt_df_3_opt2)

                st.write('---')

                st.metric(label='Score Top 3 ranked', value='{}%'.format(score_top_3_ranked_dt_opt2))
                st.metric(label='Score Top 3 unranked', value='{}%'.format(score_top_3_unranked_dt_opt2))

                st.write('---')

                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st._legacy_dataframe(podium_real_dt_opt2.style.apply(df_background_color, axis=1), height=735)

    

    # ----------------------------
    # Conclusion
    # ----------------------------
    st.write('---')
    st.markdown(
        """
        ## Récap

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

    st.write('#### Top 3')
    models_top3 = ['Rég. logistique\nTop 3 ranked', 'Rég. logistique\nTop 3 unranked', 'Foret aléatoire\nTop 3 ranked', 'Foret aléatoire\nTop 3 unranked', 'Arbre de décision\nTop 3 ranked', 'Arbre de décision\nTop 3 unranked']
    score_models_top3 = [0.18, 0.45, 0.20, 0.60, 0.28, 0.49]
    color_models_top3 = ['#54b985cf', '#4c78a8', '#54b985cf', '#e10600', '#54b985cf', '#4c78a8']

    fig_top3 = plt.figure(figsize=(14, 3.5))
    plt.bar(x=models_top3, height=score_models_top3, width=0.6, color=color_models_top3)
    plt.title('Modèles')
    plt.ylabel('Score')
    plt.ylim([0,1])

    for i in range(len(score_models_top3)):
        plt.annotate(str(score_models_top3[i]), xy=(models_top3[i], score_models_top3[i]), ha='center', va='bottom', color='#fff')

    st.pyplot(fig_top3)

    st.write('#### Podium')
    models_podium = ['Rég. logistique', 'Foret aléatoire', 'Arbre de décision']
    score_models_podium  = [0.60, 0.63, 0.65]
    color_models_podium  = [ '#4c78a8', '#4c78a8', '#e10600']

    fig_podium = plt.figure(figsize=(6.5, 3.5))
    plt.bar(x=models_podium, height=score_models_podium, width=0.6, color=color_models_podium)
    plt.title('Modèles')
    plt.ylabel('Score')
    plt.ylim([0,1])

    for i in range(len(score_models_podium)):
        plt.annotate(str(score_models_podium[i]), xy=(models_podium[i], score_models_podium[i]), ha='center', va='bottom', color='#fff')

    col1_results, col2_results = st.columns(2)
    with col1_results:
        st.pyplot(fig_podium)
    
    st.markdown(
        """
        #### Top 3 + Podium

        - Option 1
        """)
    models_top3_podium_opt1 = ['Rég. logistique\nTop 3 ranked', 'Rég. logistique\nTop 3 unranked', 'Foret aléatoire\nTop 3 ranked', 'Foret aléatoire\nTop 3 unranked', 'Arbre de décision\nTop 3 ranked', 'Arbre de décision\nTop 3 unranked']
    score_models_top3_podium_opt1 = [0.35, 0.60, 0.28, 0.63, 0.18, 0.65]
    color_models_top3_podium_opt1 = ['#54b985cf', '#4c78a8', '#54b985cf', '#4c78a8', '#54b985cf', '#e10600']

    fig_top3_podium_opt1 = plt.figure(figsize=(14, 3.5))
    plt.bar(x=models_top3_podium_opt1, height=score_models_top3_podium_opt1, width=0.6, color=color_models_top3_podium_opt1)
    plt.title('Modèles')
    plt.ylabel('Score')
    plt.ylim([0,1])

    for i in range(len(score_models_top3_podium_opt1)):
        plt.annotate(str(score_models_top3_podium_opt1[i]), xy=(models_top3_podium_opt1[i], score_models_top3_podium_opt1[i]), ha='center', va='bottom', color='#fff')
    
    st.pyplot(fig_top3_podium_opt1)

    st.markdown(
        """
        - Option 2
        """)
    models_top3_podium_opt2 = ['Rég. logistique\nTop 3 ranked', 'Rég. logistique\nTop 3 unranked', 'Foret aléatoire\nTop 3 ranked', 'Foret aléatoire\nTop 3 unranked', 'Arbre de décision\nTop 3 ranked', 'Arbre de décision\nTop 3 unranked']
    score_models_top3_podium_opt2 = [0.22, 0.62, 0.21, 0.62, 0.38, 0.69]
    color_models_top3_podium_opt2 = ['#54b985cf', '#4c78a8', '#54b985cf', '#4c78a8', '#54b985cf', '#e10600']

    fig_top3_podium_opt2 = plt.figure(figsize=(14, 3.5))
    plt.bar(x=models_top3_podium_opt2, height=score_models_top3_podium_opt2, width=0.6, color=color_models_top3_podium_opt2)
    plt.title('Modèles')
    plt.ylabel('Score')
    plt.ylim([0,1])

    for i in range(len(score_models_top3_podium_opt2)):
        plt.annotate(str(score_models_top3_podium_opt2[i]), xy=(models_top3_podium_opt2[i], score_models_top3_podium_opt2[i]), ha='center', va='bottom', color='#fff')
    
    st.pyplot(fig_top3_podium_opt2)
