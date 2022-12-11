import streamlit as st

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

title = "Modélisations"
sidebar_name = "Modélisation"

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

    # modif valeurs positionOrder à 1 pour prédire le gagnant (position = 1) sinon 0 pour les autres valeurs
    df['positionOrder'] = df['positionOrder'].apply(lambda x: 1 if x==1 else 0)
    st.dataframe(df.head(20))

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

    # rééchantillonnage (Régréssion Log + Arbre de décision)
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    X_ro, y_ro = ros.fit_resample(X_train_scaled, y_train)
    
    # ----------------------------
    # Algorithmes
    # ----------------------------
    st.markdown(
        """
        ## Machine learning
        """)

    model_selector = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision'),
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)
    
    if model_selector == 'Régression logistique':
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            C_param_selector = st.selectbox(label='C', options=(0.001, 0.01, 0.1, 1, 10), index=2)

        if st.button('Résultats'):  

            st.write('---')

            # instanciation modèle
            log_reg = LogisticRegression(C=C_param_selector)
            log_reg.fit(X_ro, y_ro)

            # probabilité avec predict_proba
            y_pred_log_ro2 = log_reg.predict_proba(X_test_scaled)
            df_y_pred_log_ro2 = pd.DataFrame(y_pred_log_ro2, columns=['proba_0', 'proba_1'])
            #st.dataframe(df_y_pred_log_ro2.head(20))


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

            classif_report_df = pd.DataFrame(classification_report(y_test, df_test_proba['prediction'], output_dict=True)).T[:2]
            classif_report_df['support'] = classif_report_df['support'].astype('int')
            
            col1, col2 = st.columns(2)
            with col1:
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


            with col2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_log, height=735)
    

    elif model_selector == 'Forêt aléatoire':
        # ----------------------
        # Modèle Forêt aléatoire
        # ----------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            n_estimators_param_selector = st.selectbox(label='n_estimators', options=(10, 50, 100, 250), index=2)
        with param_col2:
            min_samples_leaf_param_selector = st.selectbox(label='min_samples_leaf', options=(1, 3, 5), index=0)
        with param_col3:
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1)

        if st.button('Résultats'):

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

            classif_report_rf_df = pd.DataFrame(classification_report(y_test, df_test_prob_rf['prediction'], output_dict=True)).T[:2]
            classif_report_rf_df['support'] = classif_report_rf_df['support'].astype('int')
            
            col1, col2 = st.columns(2)
            with col1:
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


            with col2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_rf, height=735)
    

    elif model_selector == 'Arbre de décision':
        # ------------------------
        # Modèle Arbre de décision
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            criterion_param_selector = st.selectbox(label='criterion', options=('entropy', 'gini'), index=0)
        with param_col2:
            mmax_depth_param_selector = st.selectbox(label='max_depth', options=(1, 2, 3, 5, 6, 7), index=3)

        if st.button('Résultats'):

            st.write('---')

            # instanciation modèle
            dt_clf = DecisionTreeClassifier(criterion=criterion_param_selector, max_depth=5, random_state=143)
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

            classif_report_dt_df = pd.DataFrame(classification_report(y_test, df_test_prob_dt['prediction'], output_dict=True)).T[:2]
            classif_report_dt_df['support'] = classif_report_dt_df['support'].astype('int')
            
            col1, col2 = st.columns(2)
            with col1:
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


            with col2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_dt, height=735)

    