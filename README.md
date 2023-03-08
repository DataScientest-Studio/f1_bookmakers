# AVR22 CDS : F1 Bookmakers



## Presentation


This repository contains the code for our project **F1 & Bookmakers**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to **beat the bookmaker during a F1 season, predicting either the winner or the podium of each race of a F1 season.** In addition, we will also try to predict the overall standings for the 2021 season.

This project was developed by the following team :

- Sébastien Lebreton ([GitHub](https://github.com/kente92) / [LinkedIn](https://www.linkedin.com/in/seblebreton/))
- Alexandre Laroche ([GitHub](https://github.com/Alex-Laroche) / [LinkedIn](https://www.linkedin.com/in/alexandre-laroche-a96360263/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :


```
pip install -r requirements.txt
```

## Streamlit App

To run the app :

```shell
cd streamlit_app
conda create --name f1-bookmaker-streamlit python=3.9
conda activate f1-bookmaker-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
