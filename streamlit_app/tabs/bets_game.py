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

    resultats_vainqueurs_2021 = pd.read_csv(r"../data/resultats_vainqueurs_2021.csv", sep=';', decimal=',')
    resultats_vainqueurs_2021 = resultats_vainqueurs_2021.merge(right=races[['round', 'name']], on='round')

    resultats_top3_2021 = pd.read_csv(r"../data/resultats_top3_2021.csv", sep=';')
    resultats_top3_2021 = resultats_top3_2021.merge(right=races[['round', 'name']], on='round')


    # st.write('Résultats Vainqueurs')
    # st._legacy_dataframe(resultats_vainqueurs_2021)
    # st.write('Résultats Top3')
    # st._legacy_dataframe(resultats_top3_2021)
    # st.write('Races 2021')
    # st._legacy_dataframe(races)


    bet_selector = st.selectbox(label='Quel pari ?', options=('', 'Pari vainqueur', 'Pari Top 3'), key="bets",
                                    format_func=lambda x: "< Choix du pari >" if x == '' else x)
    
    if bet_selector == 'Pari vainqueur':

        grand_prix_winner_selector = st.selectbox(label='Choix de la course', options=circuits_list, index=0, key='grand_prix_winner')

        if st.checkbox('Voir les prédictions', key='winner_bet_result'):  

            #st.dataframe(resultats_vainqueurs_2021[resultats_vainqueurs_2021['name']==grand_prix_winner_selector])

            grand_prix_pred_df = resultats_vainqueurs_2021[resultats_vainqueurs_2021['name']==grand_prix_winner_selector].reset_index(drop=True)

            predicted_winner_log_reg_name = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[0,'Predicted winner']
            predicted_winner_log_reg_cote = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[0,'Cote']
            predicted_winner_rf_name = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[0,'Predicted winner']
            predicted_winner_rf_cote = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[0,'Cote']
            predicted_winner_dt_name = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[0,'Predicted winner']
            predicted_winner_dt_cote = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[0,'Cote']
    
            result_winner_col1, result_winner_col2, result_winner_col3 = st.columns(3)

            with result_winner_col1:
                st.write('#### Régression logistique')
                st.write('Pilote :', predicted_winner_log_reg_name)
                st.write('Cote :', str(predicted_winner_log_reg_cote))
            
            with result_winner_col2:
                st.write('#### Fôret aléatoire')
                st.write('Pilote :', predicted_winner_rf_name)
                st.write('Cote :', str(predicted_winner_rf_cote))
            
            with result_winner_col3:
                st.write('#### Arbre de décision')
                st.write('Pilote :', predicted_winner_dt_name)
                st.write('Cote :', str(predicted_winner_dt_cote))
            
            unique_drivers = grand_prix_pred_df[['Winner', 'Predicted winner', 'Cote']].drop_duplicates()
            #st.write(unique_drivers)
            
            select_driver_col1, select_bet_col2 = st.columns(2)
            with select_driver_col1:
                predicted_winners_selector = st.selectbox(label='Quel pilote parier ?',
                                                            options=['']+list(unique_drivers['Predicted winner']),
                                                            index=0, key='predicted_winners',
                                                            format_func=lambda x: "< Choix du pilote >" if x == '' else x)

            with select_bet_col2:
                bet_amount = st.number_input('Mise')
            
            if (predicted_winners_selector in list(unique_drivers['Predicted winner'])) & (bet_amount!=0):
                if st.button('Résultat', key='winner_bet'):
                    
                    true_winner = unique_drivers[unique_drivers['Predicted winner']==predicted_winners_selector].reset_index(drop=True).loc[0,'Winner']
                    selected_driver_cote = unique_drivers[unique_drivers['Predicted winner']==predicted_winners_selector].reset_index(drop=True).loc[0,'Cote']

                    if predicted_winners_selector == true_winner:
                        st.write('Bravo, vous avez gagné le pari')
                        st.write('Gain :', str((selected_driver_cote * bet_amount) - bet_amount))
                    else:
                        st.write(
                            f"""
                            Dommage, vous avez perdu le pari

                            Le vainqueur est : {true_winner}
                            """)
                        st.write('Perte :', str(- bet_amount))

    elif bet_selector == 'Pari Top 3':

        st.write('Top 3 - à faire')

