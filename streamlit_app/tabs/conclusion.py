import streamlit as st


title = "Conclusion"
sidebar_name = "Conclusion"

def run():
    
    st.title(title)

    st.markdown(
        """
        

        ## Objectif du projet

        Le but du projet était de prédire le gagnant et le top 3 des courses de Formule 1, de parier et voir si nos pronostics étaient gagnants. 
        Nous avons même poussé l'exercice jusqu'à prédire l'intégralité du classement sur une saison complète.

        ---

        ## Résumé

        #### Vainqueur de la course
        
        - 5 modèles : Régression Logistique, Forêt aléatoire, Arbre de décision, SVC, KNN
        - f1-score : de 0.31 (KNN) à 0.55 (régression logistique)
        - ROI des paris associés : :red[-10,5 % (forêt aléatoire)] à :green[16 % (régression logistique)]
        - _Rappel : le ROI moyen pour des paris en France : -17%_
        - _Les parieurs "pro" annoncent un ROI de 10%_
        
        ---
        #### Top 3
        
        - 3 modèles : Régression Logistique, Forêt aléatoire, Arbre de décision
        - f1-score : de 0.62 (forêt aléatoire & régression logistique) à 0.69 (arbre de décision)
        - ROI des paris associés :  :red[-10 % (régression logistique)], :green[48 % (forêt aléatoire)] et :green[**72 % (arbre de décision)**]
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
        - Ecart de points moyen : :green[10.5 (Arbre de décision)] à :red[88 (LinearSVC)]
        - Ecart de classement moyen :  :green[0.4 (Arbre de décision)] à :red[4 (LinearSVC)]
        
        ---

        ## Perspectives et évolutions possibles

        L'utilisation de SHAP nous a montré que certaines features n'étaient pas pertinentes.
        - Supprimer les données météo sauf l'info "pluie".
        - La classe du circuit n'a pas d'impact en l'état.
        
        Compléter le jeu de données :
        - Echanger avec un professionnel du métier évidemment.
        - Utiliser la libraire fastf1 pour l'enrichir (vitesse max, vitesse moyenne dans les virages, régularité du pilote, etc.).
          - La classe du circuit aura peut-être alors un intérêt.
        - Créer un indicateur "forme du moment" ou "efficacité de la stratégie".
        - Budget de l'écurie ou le salaire du pilote.
        - Nombre d'apparitions de la Safety Car sur le circuit.
        - ...
        
        ---
        
        > **"En Formule 1, chance et malchance n'existent pas.**
        > **Cette dernière n'est autre que la somme d'éléments ou de situations que nous n'avons pas su ou pu prévoir",** _Enzo Ferrari_

        ---
        
        """, unsafe_allow_html=True
    )