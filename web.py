import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽️",
    layout="wide",
    initial_sidebar_state="expanded")

# Sidebar menu
st.sidebar.header('Pages')
page = st.sidebar.selectbox('', ['Team Analysis Tool', 'Comparison Tool', 'Info'])

if page == 'Team Analysis Tool':
    st.title('Team Analysis Tool')
    df = pd.read_csv('team-data/big-ten-combined-data.csv')
    
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')

    # Filter Team
    team_ids = df['team'].unique()
    selected_team_id = st.selectbox('Select a Team ID:', team_ids)
    df_team = df[df['team'] == selected_team_id]

    # Extract the year from the date column for the selected team
    df_team['year'] = df_team['date'].dt.year

    # Define the range of seasons (0-4)
    # Assuming that the seasons are from 2021 (Season 0) to 2025 (Season 4)
    available_seasons = {"2021 Season": 2021, "2022 Season": 2022, "2023 Season": 2023, "2024 Season": 2024, "2025 Season": 2025}
    
    # Show the available seasons based on years available for the selected team
    team_years = sorted(df_team['year'].unique())
    available_seasons = {key: value for key, value in available_seasons.items() if value in team_years}
    
    # Multiselect for selecting multiple seasons
    selected_seasons_index = st.multiselect(
        'Select Seasons (2021-2024 Season, multiple selections allowed):', list(available_seasons.keys()), default=["2024 Season"]
    )

    # Map selected season indices to the corresponding years
    selected_seasons_years = [available_seasons[season_idx] for season_idx in selected_seasons_index]

    # Display the selected seasons in "xxx Season" format
    selected_seasons_display = sorted([f"{year} Season" for year in selected_seasons_years], reverse=True)
    if not selected_seasons_display:
        st.warning("Not selected any season yet, please select one to generate the report.")
    else:
        st.write(f"Selected Seasons: {', '.join(selected_seasons_display)}")


    # Filter the data based on the selected seasons (years)
    df_season = df_team[df_team['year'].isin(selected_seasons_years)]

    # Calculate averages for the selected seasons and team
    avg_goals_scored = df_season['goals'].mean()
    avg_goals_conceded = df_season['goals_against'].mean()
    passing_accuracy = df_season['pass_success_rate'].mean()
    offensive_duel_win_rate = df_season['offdwr'].mean() 
    defensive_duel_win_rate = df_season['defdwr'].mean()

    # Display the calculated stats
    st.write(f"Average Goals Scored: {avg_goals_scored}")
    st.write(f"Average Goals Conceded: {avg_goals_conceded}")
    st.write(f"Passing Accuracy: {passing_accuracy}%")
    st.write(f"Offensive Duel Win Rate: {offensive_duel_win_rate}%")
    st.write(f"Defensive Duel Win Rate: {defensive_duel_win_rate}%")



    # Extract unique match IDs
    # match_ids = df_team['match'].unique()
    # selected_match_ids = st.selectbox('Select a Match ID:', match_ids)

    # # Filter DataFrame by match ID
    # df = df_team[df_team['match'] == selected_match_ids]





    