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
sidebar_name = "modelisation"

def run():
    st.title(title)

    # -----------------------
    # préparation des données
    # -----------------------

    st.markdown('## Préparation des données')

    # chargement données
    df = pd.read_csv(r"../data/df_results_meteo_circuit_classement.csv", sep=';', index_col=0)

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


    # ----------------------------
    # Modèle régression logistique
    # ----------------------------

    st.markdown(
        """
        ## Régression logistique

        #### Modèle et prédiction avec les probabilités
        """)


    # rééchantillonnage
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    X_ro, y_ro = ros.fit_resample(X_train_scaled, y_train)

    # instanciation modèle
    log_reg = LogisticRegression(C=0.01)
    log_reg.fit(X_ro, y_ro)

    # probabilité avec predict_proba
    y_pred_log_ro2 = log_reg.predict_proba(X_test_scaled)
    df_y_pred_log_ro2 = pd.DataFrame(y_pred_log_ro2, columns=['proba_0', 'proba_1'])
    st.dataframe(df_y_pred_log_ro2.head(20))

    st.markdown("""#### Résultats""")

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
    confusion_matrix = pd.crosstab(df_test_proba['positionOrder'], df_test_proba['prediction'], rownames=['Classe réelle'], colnames=['Classe prédite'])
    st.dataframe(confusion_matrix)

    st.markdown(classification_report(y_test, df_test_proba['prediction']))

    st.markdown("""#### Comparaison pilotes vainqueurs et prédictions""")

    # chargement données pilotes
    drivers_data = pd.read_csv(r"../data/drivers.csv")

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
    st.dataframe(winner_real_log.merge(right=winner_predicted_log, on='round').sort_values(by=['round']), height=735)