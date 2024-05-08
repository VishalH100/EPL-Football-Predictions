import pandas as pd
import streamlit as st

import constants as c

# Set wide mode by default
st.set_page_config(layout='wide')

# Load the CSV data
data = pd.read_csv(c.GB_pred)

# Read Final Standings CSV file
standings_data = pd.read_csv(c.final_standings)

# Sidebar filters
st.sidebar.title("Filters")

# Function to redraw filter options
def redraw_filters():
    with st.sidebar.expander("Filter Options", expanded=True):
        selected_week = st.selectbox("Week", ["All"] + sorted(data['Week'].unique()), index=0, key="week_selectbox")
        selected_home_team = st.selectbox("Home Team", ["All"] + sorted(data['Home_Team'].unique()), index=0, key="home_team_selectbox")
        selected_away_team = st.selectbox("Away Team", ["All"] + sorted(data['Away_Team'].unique()), index=0, key="away_team_selectbox")
    return selected_week, selected_home_team, selected_away_team

# Display filters section expanded by default
selected_week, selected_home_team, selected_away_team = redraw_filters()

# Clear Filters button
if st.sidebar.button("Clear Filters"):
    selected_week = "All"
    selected_home_team = "All"
    selected_away_team = "All"
    # Clear the sidebar content
    st.sidebar.empty()
    # Redraw filter options
    # selected_week, selected_home_team, selected_away_team = redraw_filters()

# Create columns for predictions and standings
col1, col2 = st.columns([3, 2])

# Display predictions in col1
with col1:
    st.header("Predictions")
    for index, row in data.iterrows():
        # Filter the data based on selected filters
        if (selected_week == "All" or row['Week'] == selected_week) and \
                (selected_home_team == "All" or row['Home_Team'] == selected_home_team) and \
                (selected_away_team == "All" or row['Away_Team'] == selected_away_team):
            # Define color for the prediction result text
            prediction_result_color = "red" if row['Prediction_Result'] != 'Draw' else "black"
            # Constructing HTML for displaying match information tile
            match_html = f"""
                <div style="border-radius: 10px; border: 2px solid black; padding: 10px;">
                    <h3 style="text-align: center;">Week: {row['Week']} | Date: {row['Date']} | Time: {row['Time']}</h3>
                    <div style="color: navy; text-align: center; font-size: 40px;">
                        {row['Home_Team']} {int(row['Predicted_Home_Score'])} - {int(row['Predicted_Away_Score'])} {row['Away_Team']}
                    </div>
                    <h4 style="text-align: center; color: red ;">Prediction Result: {row['Prediction_Result']}</h4>
                    <h4 style="text-align: center;">Venue: {row['Venue']} | Referee: {row['Referee']}</h4>
                    <div style="color: green; text-align: center; font-size: 25px";>
                        Bet365 Home: {row['B365H']} | Bet365 Draw: {row['B365D']} | Bet365 Away: {row['B365A']}
                    </div>
                </div>
                """
            # Display match information tile
            st.markdown(match_html, unsafe_allow_html=True)

# Display standings in col2
with col2:
    st.header("Predicted Final Standings")
    st.write(standings_data)
