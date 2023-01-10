import streamlit as st

import pandas as pd


title = "Paris"
sidebar_name = "Paris"

def run():

    st.title(title)

    st.markdown(
        """
        Faites vos paris !
        """
    )


    races = pd.read_csv(r'../data/races.csv')
    races = races[races['year']==2021].sort_values(by='round')
    circuits_list = list(races['name'])

    resultats_vainqueurs_2021 = pd.read_csv(r"../data/resultats_vainqueurs_2021.csv", sep=';')
    resultats_vainqueurs_2021 = resultats_vainqueurs_2021.merge(right=races[['round', 'name']], on='round')

    resultats_top3_2021 = pd.read_csv(r"../data/resultats_top3_2021.csv", sep=';')
    resultats_top3_2021 = resultats_top3_2021.merge(right=races[['round', 'name']], on='round')


    st.write('Résultats Vainqueurs')
    st._legacy_dataframe(resultats_vainqueurs_2021)
    st.write('Résultats Top3')
    st._legacy_dataframe(resultats_top3_2021)
    st.write('Races 2021')
    st._legacy_dataframe(races)


    grand_prix_selector = st.selectbox(label='Course', options=circuits_list, index=0, key='grand_prix')

    if st.button('Résultats', key='log-iter1'):  

            st.dataframe(resultats_vainqueurs_2021[resultats_vainqueurs_2021['name']==grand_prix_selector])

