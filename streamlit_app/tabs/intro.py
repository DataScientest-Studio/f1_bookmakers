import streamlit as st


title = "F1 bookmaker"
sidebar_name = "Introduction"


def run():

    st.image("https://www.formula1.com/content/dam/fom-website/sutton/2022/Italy/Sunday/1422823415.jpg.transform/9col/image.jpg", width=704)

    st.title(title)

    st.markdown(
        """
        Ce projet a été réalisé dans le cadre de notre formation Data Scientist

        Le but de ce projet est de se servir de l’ensemble des données disponibles pour prédire soit le gagnant, soit le podium de chaque course d’une saison de F1. En complément nous essayerons aussi de prédire le classement général sur la saison 2021.
        """
    )
