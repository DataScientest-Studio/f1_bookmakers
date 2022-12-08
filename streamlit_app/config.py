"""

Config file for Streamlit App

"""

from member import Member


TITLE = "Streamlit Formula 1"

TEAM_MEMBERS = [
    Member(
        name="John Doe",
        linkedin_url="https://www.linkedin.com/in/charlessuttonprofile/",
        github_url="https://github.com/charlessutton",
    ),
    Member("Jane Doe"),
]

PROMOTION = "Promotion Continue Data Scientist - Avril 2022"
