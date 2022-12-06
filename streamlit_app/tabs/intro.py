import streamlit as st


title = "F1 bookmaker"
sidebar_name = "Introduction"

def run():

    st.markdown('<style>section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] {border-top: 8px solid var(--red-color); border-right: 8px solid var(--red-color); border-top-right-radius: 23px;} section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] img {border-top-right-radius: 15px;} button[title="View fullscreen"] {display: none;}</style>', unsafe_allow_html=True)
    st.image(r"./assets/banniere_intro.jpg", width=1080)

    st.title(title)

    st.markdown(
        """
        Ce projet a été réalisé dans le cadre de notre formation Data Scientist

        Le but de ce projet est de se servir de l’ensemble des données disponibles pour prédire soit le gagnant, soit le podium de chaque course d’une saison de F1. En complément nous essayerons aussi de prédire le classement général sur la saison 2021.
        """
    )
