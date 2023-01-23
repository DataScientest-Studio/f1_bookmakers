"""

Config file for Streamlit App

"""

from member import Member


TITLE = "Streamlit Formula 1"

TEAM_MEMBERS = [
    Member(
        name="Seb",
        linkedin_url="https://www.linkedin.com/in/seblebreton/",
        github_url="https://github.com/kente92",
    ),
    Member("Alexandre", "https://www.linkedin.com/in/alexandre-laroche-a96360263/", "https://github.com/Alex-Laroche"),
]

PROMOTION = "Promotion Continue<br>Data Scientist - Avril 2022"
