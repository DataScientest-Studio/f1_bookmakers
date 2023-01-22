import streamlit as st

import pandas as pd


title = "Faites vos paris !"
sidebar_name = "Bonus : pariez !"

def run():

    st.markdown('<style>section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] {border-top: 5px solid var(--red-color); border-right: 5px solid var(--red-color); border-top-right-radius: 20px;} section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] img {border-top-right-radius: 15px;} .stMultiSelect div div div div div:nth-of-type(2) {visibility: hidden;} .stMultiSelect div div div div div:nth-of-type(2)::before {visibility: visible; content:"Choisissez 3 pilotes"} </style>', unsafe_allow_html=True)

    st.title(title)


    races = pd.read_csv(r'../data/races.csv')
    races = races[races['year']==2021].sort_values(by='round')

    resultats_vainqueurs_2021 = pd.read_csv(r"../data/resultats_vainqueurs_2021.csv", sep=';', decimal=',')
    resultats_vainqueurs_2021 = resultats_vainqueurs_2021.merge(right=races[['round', 'name']], on='round')

    resultats_top3_2021 = pd.read_csv(r"../data/resultats_top3_2021.csv", sep=';', decimal=',')
    resultats_top3_2021 = resultats_top3_2021.merge(right=races[['round', 'name']], on='round')

    circuits_list = list(resultats_vainqueurs_2021['name'].drop_duplicates())

    # st.write('Résultats Vainqueurs')
    # st._legacy_dataframe(resultats_vainqueurs_2021)
    # st.write('Résultats Top3')
    # st._legacy_dataframe(resultats_top3_2021)
    # st.write('Races 2021')
    # st._legacy_dataframe(races)


    bet_selector = st.selectbox(label='', options=('', 'Pari vainqueur', 'Pari Top 3'), key="bets",
                                    format_func=lambda x: "< Choisissez un pari >" if x == '' else x)
    
    if bet_selector == 'Pari vainqueur':

        grand_prix_winner_selector = st.selectbox(label='Choix de la course', options=circuits_list, index=0, key='grand_prix_winner')

        if st.checkbox('Voir les prédictions', key='winner_bet_result'):  

            # st.dataframe(resultats_vainqueurs_2021[resultats_vainqueurs_2021['name']==grand_prix_winner_selector])

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
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_winner_log_reg_name), width=260)
                st.write('Pilote :', predicted_winner_log_reg_name)
                st.write('Cote :', str(predicted_winner_log_reg_cote))
            
            with result_winner_col2:
                st.write('#### Fôret aléatoire')
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_winner_rf_name), width=260)
                st.write('Pilote :', predicted_winner_rf_name)
                st.write('Cote :', str(predicted_winner_rf_cote))
            
            with result_winner_col3:
                st.write('#### Arbre de décision')
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_winner_dt_name), width=260)
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

        grand_prix_top3_selector = st.selectbox(label='Choix de la course', options=circuits_list, index=0, key='grand_prix_top3')

        if st.checkbox('Voir les prédictions', key='top3_bet_result'):  

            #st.dataframe(resultats_top3_2021[resultats_top3_2021['name']==grand_prix_top3_selector])

            grand_prix_pred_df = resultats_top3_2021[resultats_top3_2021['name']==grand_prix_top3_selector].reset_index(drop=True)

            predicted_top3_log_reg_name1 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[0,'Predicted driver']
            predicted_top3_log_reg_cote1 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[0,'Cote Top 3']
            predicted_top3_log_reg_name2 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[1,'Predicted driver']
            predicted_top3_log_reg_cote2 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[1,'Cote Top 3']
            predicted_top3_log_reg_name3 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[2,'Predicted driver']
            predicted_top3_log_reg_cote3 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Logistic regression'].reset_index(drop=True).loc[2,'Cote Top 3']
            predicted_top3_rf_name1 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[0,'Predicted driver']
            predicted_top3_rf_cote1 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[0,'Cote Top 3']
            predicted_top3_rf_name2 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[1,'Predicted driver']
            predicted_top3_rf_cote2 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[1,'Cote Top 3']
            predicted_top3_rf_name3 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[2,'Predicted driver']
            predicted_top3_rf_cote3 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Random forest'].reset_index(drop=True).loc[2,'Cote Top 3']
            predicted_top3_dt_name1 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[0,'Predicted driver']
            predicted_top3_dt_cote1 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[0,'Cote Top 3']
            predicted_top3_dt_name2 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[1,'Predicted driver']
            predicted_top3_dt_cote2 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[1,'Cote Top 3']
            predicted_top3_dt_name3 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[2,'Predicted driver']
            predicted_top3_dt_cote3 = grand_prix_pred_df[grand_prix_pred_df['Model']=='Decision tree'].reset_index(drop=True).loc[2,'Cote Top 3']

            st.write('#### Régression logistique')
            result_top3_log_reg_col1, result_top3_log_reg_col2, result_top3_log_reg_col3 = st.columns(3)
            with result_top3_log_reg_col1:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_log_reg_name1), width=260)
                st.write('Pilote 1 :', predicted_top3_log_reg_name1)
                st.write('Cote :', str(predicted_top3_log_reg_cote1))
            with result_top3_log_reg_col2:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_log_reg_name2), width=260)
                st.write('Pilote 2 :', predicted_top3_log_reg_name2)
                st.write('Cote :', str(predicted_top3_log_reg_cote2))
            with result_top3_log_reg_col3:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_log_reg_name3), width=260)
                st.write('Pilote 3 :', predicted_top3_log_reg_name3)
                st.write('Cote :', str(predicted_top3_log_reg_cote3))
            
            st.write('---')

            st.write('#### Fôret aléatoire')
            result_top3_rf_col1, result_top3_rf_col2, result_top3_rf_col3 = st.columns(3)
            with result_top3_rf_col1:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_rf_name1), width=260)
                st.write('Pilote 1 :', predicted_top3_rf_name1)
                st.write('Cote :', str(predicted_top3_rf_cote1))
            with result_top3_rf_col2:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_rf_name2), width=260)
                st.write('Pilote 2 :', predicted_top3_rf_name2)
                st.write('Cote :', str(predicted_top3_rf_cote2))
            with result_top3_rf_col3:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_rf_name3), width=260)
                st.write('Pilote 3 :', predicted_top3_rf_name3)
                st.write('Cote :', str(predicted_top3_rf_cote3))
            
            st.write('---')

            st.write('#### Arbre de décision')
            result_top3_dt_col1, result_top3_dt_col2, result_top3_dt_col3 = st.columns(3)
            with result_top3_dt_col1:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_dt_name1), width=260)
                st.write('Pilote 1 :', predicted_top3_dt_name1)
                st.write('Cote :', str(predicted_top3_dt_cote1))
            with result_top3_dt_col2:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_dt_name2), width=260)
                st.write('Pilote 2 :', predicted_top3_dt_name2)
                st.write('Cote :', str(predicted_top3_dt_cote2))
            with result_top3_dt_col3:
                st.image(r'./assets/Drivers2021/{}.jpg'.format(predicted_top3_dt_name3), width=260)
                st.write('Pilote 3 :', predicted_top3_dt_name3)
                st.write('Cote :', str(predicted_top3_dt_cote3))
            
            st.write('---')
            top3_predicted_drivers = grand_prix_pred_df[['Predicted driver', 'Cote Top 3']].drop_duplicates()
            #st.write(top3_predicted_drivers)

            top3_drivers = grand_prix_pred_df['Driver'].unique()
            #st.write(list(top3_drivers))

            select_top3_drivers_col1, select_top3_bet_col2 = st.columns(2)
            with select_top3_drivers_col1:
                drivers_selection = st.multiselect(label='Quels pilotes parier ?', options=top3_predicted_drivers, key='bet_top3_selection')

            with select_top3_bet_col2:
                bet_top3_amount = st.number_input('Mise pour chaque pilote')

            if len(drivers_selection) > 3:
                st.warning("You have to select only 3 drivers")
            
            elif (len(drivers_selection) == 3) & (bet_top3_amount!=0):
                #st.write(drivers_selection)
                if st.button('Résultat', key='top_bet'):
                    total_gain = 0
                    for driver in drivers_selection:
                        if driver in top3_drivers:
                            selected_driver_top3_cote = top3_predicted_drivers[top3_predicted_drivers['Predicted driver']==driver].reset_index(drop=True).loc[0,'Cote Top 3']
                            driver_gain = (selected_driver_top3_cote * bet_top3_amount) - bet_top3_amount
                            st.write('{} : ✅'.format(driver), '   Gain :', str(driver_gain))
                            total_gain += driver_gain
                        else:
                            st.write('{} : ❌'.format(driver), '   Perte :', str(- bet_top3_amount))
                            total_gain += (- bet_top3_amount)
                    st.write('Total :', str(total_gain))


