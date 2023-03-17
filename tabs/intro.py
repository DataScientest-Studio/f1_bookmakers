import streamlit as st


title = "F1 bookmaker"
sidebar_name = "Introduction"

def run():

    st.markdown('<style> section[tabindex="0"] div[data-testid="stVerticalBlock"] > div:nth-child(3) div[data-testid="stImage"] {border-top: 8px solid var(--red-color); border-right: 8px solid var(--red-color); border-top-right-radius: 23px; margin: auto;} section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] > img {border-top-right-radius: 15px;} section[tabindex="0"] div[data-testid="stVerticalBlock"] > div:nth-child(3) button[title="View fullscreen"] {display: none;} section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] {border-top: 3px solid var(--red-color); border-right: 3px solid var(--red-color); border-top-right-radius: 18px; margin: auto;}</style>', unsafe_allow_html=True)
    st.image(r"./assets/banniere_intro.jpg", width=1080)

    st.title(title)

    st.markdown(
        """
        Ce projet a été réalisé dans le cadre de notre formation Data Scientist chez DataScientest

        ## Objectif du projet

        Le but de ce projet est de se servir de l’ensemble des données disponibles pour prédire soit le gagnant, soit le podium de chaque course d’une saison de F1. En complément nous essayerons aussi de prédire le classement général sur la saison 2021.

        Ces prédictions nous permettront alors de parier en ligne face à des bookmakers et voir si nous finissons la saison bénéficiaires.

        NB : l’idée originale était d’utiliser un jeu de données de paris pour voir si le modèle permettait de gagner de l’argent, de voir à quel moment il était plus efficace de parier, etc. Malheureusement nous n’avons pas trouvé de jeu de données complet de pari permettant de mettre en place ces idées. 

        Pour les cotes présentées dans ce document, elles se basent pour la plupart sur les données de 2022 et un soupçon d’analyse personnelle.

        ---

        ## L’industrie du pari sportif en France

        Selon l’__[ARJEL](https://anj.fr/sites/default/files/2022-04/ANJ_Rapport_e%CC%81co_2021.pdf)__ , le produit brut des jeux (c.a.d. les bénéfices des bookmakers) des paris sportifs (hors courses hippiques) en France représentait 472 millions d’euros en 2017 et 1,3 milliard en 2021 pour un total de mises de 7,9 milliards d’euros (2,5 milliards en 2017)!
        Ce marché est donc en pleine croissance et les compétitions comme la coupe du monde de football 2022 et celle de rugby en France en 2023 devraient accroitre cette dynamique.

        Le total de parieurs est estimé à 4,4 millions ce qui signifie que chaque parieur mise en moyenne 1763€/an et rapporte aux organismes de pari 303€ !

        En 2021, la Formule 1 ne faisait pas partie du Top 7 des sports les plus pariés, classement évidemment dominé par le football suivi par le basket et le tennis. C’est un sport « mineur » pour les organismes de pari. Ce qui signifie que moins de moyens sont alloués pour attribuer les côtes et donc qu’il est certainement plus simple de gagner sur le long terme pour les parieurs et parieuses.

        ### Qu’est-ce qu’une cote ?

        Nous allons citer __[Wikipedia](https://fr.wikipedia.org/wiki/Pari_sportif#Cote_europ%C3%A9enne)__ qui l’explique simplement :
        « La cote européenne est utilisée comme son nom l'indique en Europe. En France par exemple, elle est proposée par défaut sur les sites de paris sportifs. C'est un indice décimal, qui indique le gain potentiel suivant la mise : **Gain = Cote * Mise**

         """, unsafe_allow_html=True
    )

    st.image(r"./assets/cotes-paris-sportifs.png", width=600)       

    st.markdown(
                """ 
        Pour un pari d'une mise de 100 € sur la victoire de A, si elle se réalise, le parieur reçoit un gain correspondant à la mise multipliée par la cote, soit 100 * 1,2 = 120 €. En tenant compte de la mise de 100 €, le parieur a donc réalisé une plus-value de 120 - 100 = 20 €.

        La cote est définie par les bookmakers comme l’inverse de la probabilité d’un évènement. Par exemple, si le bookmaker estime que l’équipe A a 63 % de chance de gagner, l’équipe B 14 % et le match nul 23% alors les cotes maximales seront : 
        - Victoire équipe A : 1/0.63 = 1.59
        - Match Nul : 1/0.23 = 4.34
        - Victoire équipe B : 1/0.14 = 7.14

        Les cotes proposées par le bookmaker sont évidemment inférieures aux cotes calculées grâce aux probabilités de victoire. 
        Pour plusieurs raisons, tout d’abord le modèle qui prédit le gagnant n’est jamais parfait et possède forcément un indice de confiance, le bookmaker va simplement prendre en compte cela dans la définition de sa cote. 
        De plus, le but du bookmaker est de gagner de l’argent et de minimiser ses pertes, il n’a aucun intérêt à établir une cote supérieure à celle calculée théoriquement (hors offres d’appel, promotions, etc.). Il doit cependant proposer des cotes « attirantes » pour inciter les joueurs à parier.

        Ainsi, pour ce même match, les cotes affichées sont de :
        - Victoire équipe A : 1.59
            - Correspond à une probabilité de victoires de 62.8 %
        - Match Nul : 4.0
            - Probabilité de victoires : 25%
        - Victoire équipe B : 5.05
            - Probabilité de victoires : 20%


        Pour la formule 1, l’idée est la même, le bookmaker établit le pourcentage de victoires de chaque pilote et calcule la cote associée.

        ---

        ## La Formule 1, une compétition mondiale

        La Formule 1 (F1) est une compétition automobile internationale avec des voitures de course monoplaces régie par la Fédération Internationale de l'Automobile (FIA). Le mot « Formule » dans le nom fait référence à l'ensemble des règles auxquelles doivent se conformer les voitures de tous les participants.

        Une saison de Formule 1 consiste en une série de courses, appelées Grands Prix, qui se déroulent dans le monde entier sur des circuits construits à cet effet.
        """, unsafe_allow_html=True
    )

    st.image(r"./assets/F1-Calendar-2021.jpg", width=700)       

    st.markdown(
        """
        Il existe deux championnats du monde annuels dans cette discipline :
        - Le championnat du monde des pilotes (saison inaugurale en 1950)
        - Le championnat du monde des constructeurs (saison inaugurale en 1958).

        Chaque constructeur F1 dispose d’une écurie avec deux pilotes et les résultats de chaque course sont évalués par un système de points. Les pilotes d'une même écurie doivent travailler en équipe pour contribuer au championnat des constructeurs, tout en étant en concurrence entre eux et avec les autres pilotes de la grille pour remporter le championnat des pilotes. Pour cette raison, la Formule 1 est à la fois un sport d'équipe et un sport individuel.
        """, unsafe_allow_html=True
    )

    st.image(r"./assets/F1-Lineup-2021.jpg", width=700)       

    st.markdown(
        """

        Un Grand Prix de F1 se déroule sur 3 jours le week-end et se compose de 3 parties : les essais, les qualifications et la course.

        <ul><li>Les essais se déroulent en 3 séances : FP1, FP2 et FP3. Ce sont des sessions d'essais libres permettant aux équipes de tester leurs voitures le vendredi et le samedi.</li>
        <br>
        <li>La séance de qualifications se déroule en 3 étapes : Q1, Q2 et Q3 (format actuel mise en place en 2006). Durant la Q1, tous les pilotes s'affrontent pour réaliser le meilleur temps au tour. Les 15 meilleurs pilotes participeront à la Q2 et les derniers pilotes seront éliminés. Même principe pour la Q2, où seuls les 10 meilleurs pilotes passeront à la Q3. C’est cette ultime session de qualification qui détermine l’ordre sur la grille de départ, le meilleur temps de la Q3 démarrera la course en pole position, le deuxième temps en seconde place, etc.</li>
        <br>
        <li>La course se déroule le dimanche. Les pilotes doivent effectuer un nombre défini de tours (300km + 1 tour) en appliquant une stratégie (arrêts aux stands, choix des pneumatiques, quantité d’essence et ravitaillement, …) pour terminer à la meilleure place possible. A l’issue de la course, les points sont attribués aux 10 premiers pilotes et les trois premiers montent sur le podium.</li></ul>
        """, unsafe_allow_html=True
    )

    st.image(r"./assets/F1-programme-grand-prix-portugal-2021.jpg", width=700)       

    st.markdown(
        """

        A l’issue de la saison, le pilote ainsi que l'écurie ayant cumulé le plus de points remportent respectivement le championnat du monde des pilotes et le championnat des constructeurs.
        """, unsafe_allow_html=True
    )

    
    col1, col2 = st.columns(2)
    with col1:
        st.image(r"./assets/F1-drivers-standing-2021.jpg", width=450)
    with col2:
        st.image(r"./assets/F1-constructors-standing-2021.jpg", width=450) 

    st.markdown(
        """

        ---

        ## La data dans le monde de la Formule 1

        Le monde de la formule 1 a toujours été à la pointe de l’innovation en inventant de nombreuses technologies que l’on retrouve aujourd’hui dans les voitures de série. On peut citer notamment l’invention de l’injection directe, du frein à disque, de l’aileron, de l’ABS, de l’antipatinage, des palettes au volant…

        Aujourd’hui cette innovation se retrouve dans l’utilisation des données. Lors de chaque course, 120 capteurs positionnés sur chaque voiture génèrent 3 Go de données et 1 500 points de données sont générés chaque seconde !

        Ces données sont utilisées pour extraire des statistiques mais aussi pour établir des prédictions de course.

        Une grande partie de ces données est rendue disponible au public à la fin de chaque week-end de Grand Prix par l’intermédiaire du groupement d’entreprises « Formula 1 » qui est responsable de la promotion du Championnat du monde FIA de Formule 1.


        """, unsafe_allow_html=True
    )

