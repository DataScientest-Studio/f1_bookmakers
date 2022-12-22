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
        ## Prédiction podium

        Nous souhaitons explorer la possibilité de prédire le top 3 d’arrivée d’une course.

        Nous avons opté pour une variable cible « podium » qui a pour valeur 1 les positions 1 / 2 / 3 de la variable « positionOrder » et zéro pour les autres positions.

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
                
                max_proba_2 = df_temp[df_temp['prediction']==0]['proba_2'].max()    # on identifie la valeur max proba 2 en filtrant la ligne max proba 1
                index_max_proba_2 = df_temp[df_temp['proba_2']==max_proba_2].index  # on récupère l'index de la valeur max
                df_test_proba.loc[index_max_proba_2, 'prediction'] = 2              # on affecte valeur 2 dans prediction à l'index max
                
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
            
            podium_real_log['match Top 3 ranked'] = podium_real_log.apply(lambda row: '          ✅' if row['Driver']==row['Predicted driver'] else '          ❌', axis=1)

            # podium_real_log['match'] = 0
            temp_df = podium_real_log[podium_real_log['round']==2]
            temp_df['match'] = 0

            def check_top3_unranked(df):
                drivers_list = list(df['Driver'])
                for driver in df['Predicted driver']:
                    if driver in drivers_list:
                        index_driver = df[df['Predicted driver']==driver].index
                        df.loc[index_driver, 'match'] = 1
                
                return df
            
            temp_df = check_top3_unranked(temp_df)
            #temp_df = podium_real_log[podium_real_log['round']==2]

            def df_background_color(s):
                return ['background-color: #202028']*len(s) if (s['round']%2)==0 else ['background-color: #0e1117']*len(s)

            with col2_iter1:
                st.dataframe(temp_df)
                st.write('---')
                st.markdown("""#### Pilotes top 3 VS prédictions""")
                st.dataframe(podium_real_log.style.apply(df_background_color, axis=1), height=735)
    
    elif model_selector == 'Forêt aléatoire':
        st.write('random forest')
    
    elif model_selector == 'Arbre de décision':
        st.write('tree decision')