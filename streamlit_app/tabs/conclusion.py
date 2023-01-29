import streamlit as st


title = "Conclusion"
sidebar_name = "Conclusion"

def run():

    st.title(title)

    st.markdown(
        """
        
        ---
        
        > **"En Formule 1, chance et malchance n'existent pas.**
        > **Cette dernière n'est autre que la somme d'éléments ou de situations que nous n'avons pas su ou pu prévoir",** _Enzo Ferrari_

        ---
        
        
        ## Objectif du projet

        Le but du projet était de prédire le gagnant et le top 3 des courses de Formule 1, de parier et voir si nos pronostics étaient gagnants. 
        Nous avons même poussé l'exercice jusqu'à prédire l'intégralité du classement sur une saison complète.

        ---

        ## Résumé

        #### Vainqueur de la course
        
        - 5 modèles : Régression Logistique, Forêt aléatoire, Arbre de décision, SVC, KNN
        - f1-score : de 0.31 (KNN) à 0.55 (régression logistique)
        - ROI des paris associés : <span style="color:#c10909">-10,5 % (forêt aléatoire)</span> à <span style="color:#05a705">16 % (régression logistique)</span>
        - _Rappel : le ROI moyen pour des paris en France : -17%_
        - _Les parieurs "pro" annoncent un ROI de 10%_
        
        ---
        #### Top 3
        
        - 3 modèles : Régression Logistique, Forêt aléatoire, Arbre de décision
        - f1-score : de 0.62 (forêt aléatoire & régression logistique) à 0.69 (arbre de décision)
        - ROI des paris associés :  <span style="color:#c10909">-10 % (régression logistique)</span>, <span style="color:#05a705">48 % (forêt aléatoire)</span> et <span style="color:#05a705">**72 % (arbre de décision)**</span>
        - _Rappel : le ROI moyen pour des paris en France : -17%_
        - _Les parieurs "pro" annoncent un ROI de 10%_

        ---
        
        La différence entre deux modèles se fait surtout sur les prévisions des outsiders. Les deux premiers étant souvent correctement prédits.
        
        ---
        
        #### Classement complet sur une saison
        
        Cette partie relève plus du test qu'autre chose, en France on ne peut pas parier sur un classement final complet. On peut uniquement parier sur
        le vainqueur ou des "faces à faces" (est-ce que le pilote 1 finira devant le pilote 2?) et nous n'avons pas pu obtenir les côtes associées
        à ces paris.
        
        - 4 modèles : Régression Logistique, Forêt aléatoire, Arbre de décision, Linear SVC
        - Ecart de points moyen : <span style="color:#05a705">10.5 (Arbre de décision)</span> à <span style="color:#c10909">88 (LinearSVC)</span>
        - Ecart de classement moyen :  <span style="color:#05a705">0.4 (Arbre de décision)</span> à <span style="color:#c10909">4 (LinearSVC)</span>
        
        ---

        ## Perspectives et évolutions possibles

        L'utilisation de SHAP nous a montré que certaines features n'étaient pas pertinentes.
        - Supprimer les données météo sauf l'info "pluie".
        - La classe du circuit n'a pas d'impact en l'état.
        
        Compléter le jeu de données :
        - Echanger avec un professionnel du métier.
        - Utiliser la librairie fastf1 pour l'enrichir (vitesse max, vitesse moyenne dans les virages, régularité du pilote, etc.).
          - La classe du circuit aura peut-être alors un intérêt.
        - Créer un indicateur "forme du moment" ou "efficacité de la stratégie".
        - Budget de l'écurie ou le salaire du pilote.
        - Différencier pilote principal d'une écurie et deuxième pilote.
        - Probabilité d'apparitions de la Safety Car sur le circuit.
        - ...
        
        Concernant les modèles :
        - Essayer d'autres modèles évidemment.
        - Utiliser une combinaison de plusieurs modèles.
        - Utiliser un réseau de neurones.
        - ...


        Ce projet semble montrer que l'utilisation de machine learning est une piste intéressante pour battre les bookmakers dans le domaine de la F1.
        
        
        """, unsafe_allow_html=True
    )