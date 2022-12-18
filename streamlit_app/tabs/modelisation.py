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
sidebar_name = "Modélisation - Vainqueur"

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
        ## Prédiction vainqueur
        ### Itération 1

        Nous avons ciblé le championnat 2021 comme échantillon de test pour les prédictions.
        
        Les données de la manière suivante :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse.
        - Jeu de test : les données de l’année 2021

        """)

    model_selector = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision'), key="iter1",
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)
    
    if model_selector == 'Régression logistique':
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            C_param_selector = st.selectbox(label='C', options=(0.001, 0.01, 0.1, 1, 10), index=1, key='log-iter1')

        if st.button('Résultats', key='log-iter1'):  

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
            winners_results_log['match'] = winners_results_log.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


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
            n_estimators_param_selector = st.selectbox(label='n_estimators', options=(10, 50, 100, 250), index=2, key='rf_param1-iter1')
        with param_col2:
            min_samples_leaf_param_selector = st.selectbox(label='min_samples_leaf', options=(1, 3, 5), index=0, key='rf_param2-iter1')
        with param_col3:
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param2-iter1')

        if st.button('Résultats', key='rf-iter1'):

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
            winners_results_rf['match'] = winners_results_rf.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


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
            criterion_param_selector = st.selectbox(label='criterion', options=('entropy', 'gini'), index=0, key='dt_param1-iter1')
        with param_col2:
            max_depth_param_selector = st.selectbox(label='max_depth', options=(1, 2, 3, 5, 6, 7), index=3, key='dt_param2-iter1')

        if st.button('Résultats', key='dt-iter1'):

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
            winners_results_dt['match'] = winners_results_dt.apply(lambda row: '✅' if row['Winner']==row['Predicted winner'] else '❌', axis=1)


            with col2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(winners_results_dt, height=735)

    st.write('---')

    st.markdown(
        """
        ### Itération 2

        Après la première itération du championnat 2021, nous souhaitions voir s’il était possible d’ajouter les données des courses passées dans le jeu d’entrainement à chaque course et observer les résultats obtenus.

        Pour la première course du championnat, nous avons la répartition des données suivante :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse.
        - Jeu de test : les données de la 1ere course du championnat 2021.

        Pour la deuxième course, la répartition serait la suivante :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse + les données de la 1ere course du championnat 2021.
        - Jeu de test : les données de la 2e course du championnat 2021.

        Pour la troisième course, nous aurions :
        - Jeu d’entraînement : toutes les données jusqu’à l’année 2020 incluse + les données de la 1ere et 2e courses du championnat 2021.
        - Jeu de test : les données de la 3e course du championnat 2021.

        Et ainsi de suite.

        Les modèles sont réajustés à chaque mise à jour des jeux d’entrainements et les résultats cumulés au fur et à mesure des courses.

        """)
    
    model_selector_2 = st.selectbox(label='', options=('', 'Régression logistique', 'Forêt aléatoire', 'Arbre de décision'), key='iter2',
                                    format_func=lambda x: "< Choix du modèle >" if x == '' else x)

    if model_selector_2 == 'Régression logistique':
        # ----------------------------
        # Modèle régression logistique
        # ----------------------------

        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            C_param_selector = st.selectbox(label='C', options=(0.001, 0.01, 0.1, 1, 10), index=1, key='log-iter2')

        if st.button('Résultats', key='log-iter2'):  

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
            
            col1, col2 = st.columns(2)
            with col1:
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

            with col2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(df_winner, height=735)
    
    elif model_selector_2 == 'Forêt aléatoire':
        # ----------------------
        # Modèle Forêt aléatoire
        # ----------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            n_estimators_param_selector = st.selectbox(label='n_estimators', options=(10, 50, 100, 250), index=2, key='rf_param1-iter2')
        with param_col2:
            min_samples_leaf_param_selector = st.selectbox(label='min_samples_leaf', options=(1, 3, 5), index=0, key='rf_param2-iter2')
        with param_col3:
            max_features_param_selector = st.selectbox(label='max_features', options=('sqrt', 'log2'), index=1, key='rf_param2-iter2')

        if st.button('Résultats', key='rf-iter2'):

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
            
            col1, col2 = st.columns(2)
            with col1:
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

            with col2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(df_winner_rf, height=735)
    
    elif model_selector_2 == 'Arbre de décision':
        # ------------------------
        # Modèle Arbre de décision
        # ------------------------
        
        # Choix des paramètres
        st.markdown("""#### Paramètres""")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            criterion_param_selector = st.selectbox(label='criterion', options=('entropy', 'gini'), index=0, key='dt_param1-iter2')
        with param_col2:
            max_depth_param_selector = st.selectbox(label='max_depth', options=(1, 2, 3, 5, 6, 7), index=3, key='dt_param2-iter2')

        if st.button('Résultats', key='dt-iter2'):

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
            
            col1, col2 = st.columns(2)
            with col1:
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

            with col2:
                st.markdown("""#### Pilotes vainqueurs VS prédictions""")
                st.dataframe(df_winner_dt, height=735)