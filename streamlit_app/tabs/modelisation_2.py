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

    # rééchantillonnage (Régréssion Log + Arbre de décision)
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
        st.write('reg log')
    
    elif model_selector == 'Forêt aléatoire':
        st.write('random forest')
    
    elif model_selector == 'Arbre de décision':
        st.write('tree decision')