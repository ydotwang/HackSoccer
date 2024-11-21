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
page = st.sidebar.selectbox('', ['Team Analysis', 'Comparison between teams', 'Info'])
st.sidebar.markdown("""
    This project was developed by Vicky Xu and Yuyang Wang as part of the Northwestern University MLDS Hackathon. 
    Thank you for using it, and go Cats!
""")

if page == 'Team Analysis':
    st.title('Team Analysis')
    df = pd.read_csv('team-data/big-ten-combined-data.csv')
    
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')

    # Filter Team
    team_ids = df['team'].unique()
    selected_team_id = st.selectbox('**Select a Team ID:**', team_ids)
    df_team = df[df['team'] == selected_team_id]

    # Multiselect for selecting multiple seasons
    df_team['year'] = df_team['date'].dt.year
    available_seasons = {"2021 Season": 2021, "2022 Season": 2022, "2023 Season": 2023, "2024 Season": 2024, "2025 Season": 2025}
    
    team_years = sorted(df_team['year'].unique())
    available_seasons = {key: value for key, value in available_seasons.items() if value in team_years}
    
    selected_seasons_index = st.multiselect(
        '**Select Seasons (2021-2024 Season, multiple selections allowed):**', list(available_seasons.keys()), default=["2024 Season"]
    )
    selected_seasons_years = [available_seasons[season_idx] for season_idx in selected_seasons_index]

    selected_seasons_display = sorted([f"{year} Season" for year in selected_seasons_years], reverse=True)
    if not selected_seasons_display:
        st.warning("Not selected any season yet, please select one to generate the report.")

    # Filter the data based on the selected seasons (years)
    df_season = df_team[df_team['year'].isin(selected_seasons_years)]

    # Create three columns, one for each metric category
    col = st.columns(3)

    def plot_metric(metric, data, seasons, team_id):
        # Extract variable name and description from the metric string
        var_name = metric.split(" (")[1].strip(")") 
        description = metric.split(" (")[0]  

        if var_name not in data.columns:
            st.error(f"The selected metric '{var_name}' does not exist in the dataset. Please check your selection.")
            return

        # Prepare the data
        data['month_date'] = data['date'].dt.strftime('%b %d')  
        data_sorted = data.sort_values(by=['date'])  

        four_year_average = data[var_name].mean() # Calculate the four-year average for the selected metric

        # Create a figure
        fig = go.Figure()

        # Add time-series traces for each season
        for season_year in seasons:
            season_data = data_sorted[data_sorted['year'] == season_year]

            fig.add_trace(go.Scatter(
                x=season_data['month_date'],
                y=season_data[var_name],
                mode='lines+markers',
                name=f"{season_year} Season",
                line=dict(width=2),  
                marker=dict(size=6)  
            ))

        # Add horizontal reference line for the four-year average
        fig.add_trace(go.Scatter(
            x=data_sorted['month_date'].unique(),  # Use all unique dates for the x-axis
            y=[four_year_average] * len(data_sorted['month_date'].unique()),  # Constant y-value
            mode='lines',
            name="Four-Year Average",
            line=dict(color='grey', dash='dash')  # Customize line style and color
        ))

        fig.update_layout(
            title=f"{description.capitalize()} Over Time for {team_id}",
            xaxis_title="Month and Date",
            yaxis_title=description.capitalize(),
            xaxis=dict(
                type='category', 
                categoryorder='array', 
                categoryarray=sorted(data['month_date'].unique(), key=lambda x: pd.to_datetime(x, format='%b %d')),
                tickangle=45 
            ),
            legend_title="Season",
            colorway=px.colors.qualitative.Set1  
        )

        # Render the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
          
    def display_offensive_info():
        # Check if `df_season` exists and is not empty
        if 'df_season' not in globals() or df_season.empty:
            st.warning("No data available for the selected metric and seasons.")
            return

        st.write("### Offensive Metrics Over Time")
        tabs = st.tabs(["Shots and Conversion", "Chance Creation", "Crosses and Penetration", "Passing Effectiveness"])

        with tabs[0]:  
            st.write("#### Shots and Conversion Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Shot on Target Rate (sotr)": "Percentage of shots that hit the target",
                    "Shots outside the box on Target Rate (sobotr)": "Percentage of shots taken outside the box that were on target",
                    "Shots (shots)": "Total number of shots taken by the team",
                    "Shots on Target (sot)": "Number of shots that hit the target",
                    "Shots outside the box (shots_outside_box)": "Total number of shots taken outside the box",
                    "Shots outside the box on Target (sobot)": "Shots taken outside the box that were on target"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

        with tabs[1]:  
            st.write("#### Chance Creation Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "% of positional attacks that end with a shot (pawsr)": "Positional attacks ending with a shot",
                    "% of counters that end with a shot (countwsr)": "Counters ending with a shot",
                    "% of set pieces that ended with a shot (tspwsr)": "Set pieces ending with a shot",
                    "Corners that ended with a shot rate (cornwsr)": "Corners ending with a shot",
                    "Free kicks that ended in a shot rate (fkwsr)": "Free kicks ending with a shot",
                    "Total number of entries into the 18-yard box (box_entries)": "Entries into the 18-yard box",
                    "Entries into the 18-yard box via run (box_entries_run)": "Entries into the 18-yard box by run",
                    "Entries into the 18-yard box via cross (box_entries_cross)": "Entries into the 18-yard box by cross",
                    "Touches inside the 18-yard box (touches_in_box)": "Touches in the 18-yard box"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

        with tabs[2]: 
            st.write("#### Crosses and Penetration Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Crossing accuracy (acc_cross_rate)": "Crossing accuracy",
                    "Deep completed crosses (deep_completed_crosses)": "Deep completed crosses",
                    "Deep completed passes (deep_completed_passes)": "Deep completed passes",
                    "Box Entries (box_entries)": "Total number of entries into the 18-yard box",
                    "Offensive duel win rate (offdwr)": "Offensive duel win rate"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

        with tabs[3]:  # Passing Effectiveness
            st.write("#### Passing Effectiveness Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Total passing accuracy (pass_success_rate)": "Total passing accuracy",
                    "Total ball losses (losses)": "Total ball losses",
                    "Total ball recoveries (recoveries)": "Total ball recoveries",
                    "Total duels attempted (tot_duels)": "Total duels attempted",
                    "Forward passes success rate (succ_for_passes_rate)": "Percentage of successful forward passes",
                    "Back passes success rate (succ_back_passes_rate)": "Percentage of successful back passes",
                    "Lateral passes success rate (succ_lat_passes_rate)": "Percentage of successful lateral passes",
                    "Long passes success rate (succ_long_passes_rate)": "Percentage of successful long passes",
                    "Final third passes success rate (succ_fin_third_passes_rate)": "Percentage of successful final third passes",
                    "Progressive passes success rate (succ_prog_passes_rate)": "Percentage of successful progressive passes",
                    "Smart passes success rate (succ_smart_passes_rate)": "Percentage of successful smart passes",
                    "Throw-ins success rate (succ_throw_in_rate)": "Percentage of successful throw-ins",
                    "Average passes per possession (mean_pass_per_poss)": "Average passes per possession",
                    "Average pass length (mean_pass_len)": "Average pass length",
                    "Passes per defensive action (ppda)": "Passes per defensive action (pressing intensity)"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)
    
    def display_defensive_info():
        # Check if `df_season` exists and is not empty
        if 'df_season' not in globals() or df_season.empty:
            st.warning("No data available for the selected metric and seasons.")
            return

        st.write("### Defensive Metrics Over Time")
        tabs = st.tabs(["Defensive Solidity", "Ball Recovery", "Pressing Efficiency"])

        with tabs[0]:  
            st.write("#### Defensive Solidity Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Shots on target rate conceded (sotr_against)": "Shots on target rate conceded",
                    "Defensive duel win rate (defdwr)": "Defensive duel win rate",
                    "Aerial duels win rate (airdwr)": "Aerial duels win rate",
                    "Slide tackle success rate (slide_tackled_succ_rate)": "Slide tackle success rate"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

        with tabs[1]: 
            st.write("#### Ball Recovery Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Total recoveries of the ball (recoveries)": "Total recoveries of the ball",
                    "Total recoveries of the ball in their own third (recovery_low)": "Total recoveries of the ball in their own third",
                    "Total recoveries of the ball in the middle third (recovery_med)": "Total recoveries of the ball in the middle third",
                    "Total recoveries of the ball in the final third (recovery_high)": "Total recoveries of the ball in the final third",
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

        with tabs[2]: 
            st.write("#### Pressing Efficiency Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Pressing intensity (ppda)": "Passes per defensive action (pressing intensity)",
                    "Possession (possession)": "Possession in percentage",
                    "Average passes per possession (mean_pass_per_poss)": "Average passes per possession, indirectly related to pressing"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

    def display_other_metrics():
        # Check if `df_season` exists and is not empty
        if 'df_season' not in globals() or df_season.empty:
            st.warning("No data available for the selected metric and seasons.")
            return

        st.write("### Others")
        tabs = st.tabs(["Possession and Passing", "Discipline"])

        with tabs[0]:  
            st.write("#### Possession and Passing Metrics")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Passing accuracy (pass_success_rate)": "Total passing accuracy",
                    "Match tempo (match_tempo)": "Match tempo",
                    "Long pass rate (long_pass_rate)": "Percentage of passes count as long",
                    "Average passes per possession (mean_pass_per_poss)": "Average passes per possession",
                    "Average pass length (mean_pass_len)": "Average pass length",
                    "Average shot distance (mean_shot_dist)": "Average shot distance"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

        with tabs[1]:  
            st.write("#### Discipline")
            metric = st.selectbox(
                "Select a Metric:",
                {
                    "Total number of fouls (fouls)": "Total number of fouls",
                    "Total number of yellow cards (yellows)": "Total number of yellow cards",
                    "Total number of red cards (reds)": "Total number of red cards"
                },
            )
            plot_metric(metric, df_season, selected_seasons_years, selected_team_id)

    # Give Offensive Info
    with col[0]:
        display_offensive_info()
    # Display Defensive score
    with col[1]:
        display_defensive_info()
    # Display Other score
    with col[2]:
        display_other_metrics()

if page == 'Comparison between teams':
    st.title('Comparison Between Teams')
    df = pd.read_csv('team-data/big-ten-combined-data.csv')
    
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')

    # Filter Teams
    team_ids = df['team'].unique()
    selected_teams = st.multiselect('Select Teams for Comparison:', team_ids, default=team_ids[:2])
    if len(selected_teams) < 2:
        st.warning("Select at least two teams for comparison.")
    else:
        df_teams = df[df['team'].isin(selected_teams)]

        # Extract years and seasons
        df_teams['year'] = df_teams['date'].dt.year
        available_seasons = {f"{year} Season": year for year in sorted(df_teams['year'].unique())}
        selected_seasons_index = st.multiselect(
            'Select Seasons (multiple selections allowed):', 
            list(available_seasons.keys()), 
            default=list(available_seasons.keys())[-4:] 
        )
        selected_seasons_years = [available_seasons[season_idx] for season_idx in selected_seasons_index]

        # Check data
        if not selected_seasons_years:
            st.warning("No seasons selected. Please select at least one.")
        else:
            st.write(f"Selected Teams: {', '.join(selected_teams)}")
            st.write(f"Selected Seasons: {', '.join(selected_seasons_index)}")

            df_comparison = df_teams[df_teams['year'].isin(selected_seasons_years)]

            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)

            def plot_comparison_metric(metric, data, teams, seasons):
                var_name = metric.split(" (")[1].strip(")") 
                description = metric.split(" (")[0] 

                # Check if the variable exists in the data
                if var_name not in data.columns:
                    st.error(f"The selected metric '{var_name}' does not exist in the dataset. Please check your selection.")
                    return

                # Prepare data
                data['month_date'] = data['date'].dt.strftime('%b %d')  # Format as 'Month Day'
                data_sorted = data.sort_values(by=['date'])  # Sort by full date to ensure proper order

                # Create a figure
                fig = go.Figure()

                # Add traces for each team and season
                for team in teams:
                    team_data = data_sorted[data_sorted['team'] == team]
                    for season_year in seasons:
                        season_data = team_data[team_data['year'] == season_year]

                        fig.add_trace(go.Scatter(
                            x=season_data['month_date'],
                            y=season_data[var_name],
                            mode='lines+markers',
                            name=f"{team} ({season_year})",
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))

                # Add horizontal reference line for the four-year average
                four_year_average = data[var_name].mean()
                fig.add_trace(go.Scatter(
                    x=data_sorted['month_date'].unique(),  
                    y=[four_year_average] * len(data_sorted['month_date'].unique()), 
                    mode='lines',
                    name="Four-Year Average",
                    line=dict(color='red', dash='dash')
                ))

                fig.update_layout(
                    title=f"{description.capitalize()} Comparison Across Teams and Seasons",
                    xaxis_title="Month and Date",
                    yaxis_title=description.capitalize(),
                    xaxis=dict(
                        type='category',
                        categoryorder='array',
                        categoryarray=sorted(data['month_date'].unique(), key=lambda x: pd.to_datetime(x, format='%b %d')),
                        tickangle=45
                    ),
                    legend_title="Team and Season",
                    colorway=px.colors.qualitative.Set2
                )

                # Render the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)

            def display_offensive_comparison():
                st.write("### Offensive Metrics Comparison")
                tabs = st.tabs(["Shots and Conversion", "Chance Creation", "Crosses and Penetration", "Passing Effectiveness"])

                with tabs[0]:
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Shot on Target Rate (sotr)": "Percentage of shots that hit the target",
                            "Shots outside the box on Target Rate (sobotr)": "Percentage of shots taken outside the box that were on target",
                            "Shots (shots)": "Total number of shots taken by the team",
                            "Shots on Target (sot)": "Number of shots that hit the target",
                            "Shots outside the box (shots_outside_box)": "Total number of shots taken outside the box",
                            "Shots outside the box on Target (sobot)": "Shots taken outside the box that were on target"
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

                with tabs[1]:
                    st.write("#### Chance Creation Metrics")
                    metric = st.selectbox(
                    "Select a Metric:",
                    {
                        "% of positional attacks that end with a shot (pawsr)": "Positional attacks ending with a shot",
                        "% of counters that end with a shot (countwsr)": "Counters ending with a shot",
                        "% of set pieces that ended with a shot (tspwsr)": "Set pieces ending with a shot",
                        "Corners that ended with a shot rate (cornwsr)": "Corners ending with a shot",
                        "Free kicks that ended in a shot rate (fkwsr)": "Free kicks ending with a shot",
                        "Total number of entries into the 18-yard box (box_entries)": "Entries into the 18-yard box",
                        "Entries into the 18-yard box via run (box_entries_run)": "Entries into the 18-yard box by run",
                        "Entries into the 18-yard box via cross (box_entries_cross)": "Entries into the 18-yard box by cross",
                        "Touches inside the 18-yard box (touches_in_box)": "Touches in the 18-yard box"
                    },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

                with tabs[2]: 
                    st.write("#### Crosses and Penetration Metrics")
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Crossing accuracy (acc_cross_rate)": "Crossing accuracy",
                            "Deep completed crosses (deep_completed_crosses)": "Deep completed crosses",
                            "Deep completed passes (deep_completed_passes)": "Deep completed passes",
                            "Box Entries (box_entries)": "Total number of entries into the 18-yard box",
                            "Offensive duel win rate (offdwr)": "Offensive duel win rate"
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

                with tabs[3]: 
                    st.write("#### Passing Effectiveness Metrics")
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Total passing accuracy (pass_success_rate)": "Total passing accuracy",
                            "Total ball losses (losses)": "Total ball losses",
                            "Total ball recoveries (recoveries)": "Total ball recoveries",
                            "Total duels attempted (tot_duels)": "Total duels attempted",
                            "Forward passes success rate (succ_for_passes_rate)": "Percentage of successful forward passes",
                            "Back passes success rate (succ_back_passes_rate)": "Percentage of successful back passes",
                            "Lateral passes success rate (succ_lat_passes_rate)": "Percentage of successful lateral passes",
                            "Long passes success rate (succ_long_passes_rate)": "Percentage of successful long passes",
                            "Final third passes success rate (succ_fin_third_passes_rate)": "Percentage of successful final third passes",
                            "Progressive passes success rate (succ_prog_passes_rate)": "Percentage of successful progressive passes",
                            "Smart passes success rate (succ_smart_passes_rate)": "Percentage of successful smart passes",
                            "Throw-ins success rate (succ_throw_in_rate)": "Percentage of successful throw-ins",
                            "Average passes per possession (mean_pass_per_poss)": "Average passes per possession",
                            "Average pass length (mean_pass_len)": "Average pass length",
                            "Passes per defensive action (ppda)": "Passes per defensive action (pressing intensity)"
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)
    
            def display_defensive_comparison():
                st.write("### Defensive Metrics Comparison")
                tabs = st.tabs(["Defensive Solidity", "Ball Recovery", "Pressing Efficiency"])

                with tabs[0]:  
                    st.write("#### Defensive Solidity Metrics")
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Shots on target rate conceded (sotr_against)": "Shots on target rate conceded",
                            "Defensive duel win rate (defdwr)": "Defensive duel win rate",
                            "Aerial duels win rate (airdwr)": "Aerial duels win rate",
                            "Slide tackle success rate (slide_tackled_succ_rate)": "Slide tackle success rate"
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

                with tabs[1]: 
                    st.write("#### Ball Recovery Metrics")
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Total recoveries of the ball (recoveries)": "Total recoveries of the ball",
                            "Total recoveries of the ball in their own third (recovery_low)": "Total recoveries of the ball in their own third",
                            "Total recoveries of the ball in the middle third (recovery_med)": "Total recoveries of the ball in the middle third",
                            "Total recoveries of the ball in the final third (recovery_high)": "Total recoveries of the ball in the final third",
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

                with tabs[2]: 
                    st.write("#### Pressing Efficiency Metrics")
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Pressing intensity (ppda)": "Passes per defensive action (pressing intensity)",
                            "Possession (possession)": "Possession in percentage",
                            "Average passes per possession (mean_pass_per_poss)": "Average passes per possession, indirectly related to pressing"
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

            def display_other_comparison():
                st.write("### Other Metrics Comparison")
                tabs = st.tabs(["Possession and Passing", "Discipline"])

                with tabs[0]:  
                    st.write("#### Possession and Passing Metrics")
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Passing accuracy (pass_success_rate)": "Total passing accuracy",
                            "Match tempo (match_tempo)": "Match tempo",
                            "Long pass rate (long_pass_rate)": "Percentage of passes count as long",
                            "Average passes per possession (mean_pass_per_poss)": "Average passes per possession",
                            "Average pass length (mean_pass_len)": "Average pass length",
                            "Average shot distance (mean_shot_dist)": "Average shot distance"
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

                with tabs[1]:  
                    st.write("#### Discipline")
                    metric = st.selectbox(
                        "Select a Metric:",
                        {
                            "Total number of fouls (fouls)": "Total number of fouls",
                            "Total number of yellow cards (yellows)": "Total number of yellow cards",
                            "Total number of red cards (reds)": "Total number of red cards"
                        },
                    )
                    plot_comparison_metric(metric, df_comparison, selected_teams, selected_seasons_years)

            # Display metrics in each column
            with col1:
                display_offensive_comparison()
            with col2:
                display_defensive_comparison()
            with col3:
                display_other_comparison()

if page == 'Info':
    st.title('Info')
    st.write("""
    ### Offensive Metrics Comparison
    The offensive metrics section focuses on the attacking performance of the teams. It includes the following categories:
    """)

    with st.expander("**Shots and Conversion**: Metrics related to the shots taken and their conversion rate."):
        st.write("""
        - Shot on Target Rate (sotr)
        - Shots outside the box on Target Rate (sobotr)
        - Shots (shots)
        - Shots on Target (sot)
        - Shots outside the box (shots_outside_box)
        - Shots outside the box on Target (sobot)
        """)

    with st.expander("**Chance Creation**: Metrics on creating attacking opportunities and converting them into shots."):
        st.write("""
        - % of positional attacks that end with a shot (pawsr)
        - % of counters that end with a shot (countwsr)
        - % of set pieces that ended with a shot (tspwsr)
        - Corners that ended with a shot rate (cornwsr)
        - Free kicks that ended in a shot rate (fkwsr)
        - Total number of entries into the 18-yard box (box_entries)
        - Entries into the 18-yard box via run (box_entries_run)
        - Entries into the 18-yard box via cross (box_entries_cross)
        - Touches inside the 18-yard box (touches_in_box)
        """)

    with st.expander("**Crosses and Penetration**: Metrics that measure the success of crosses and entries into the attacking zone."):
        st.write("""
        - Crossing accuracy (acc_cross_rate)
        - Deep completed crosses (deep_completed_crosses)
        - Deep completed passes (deep_completed_passes)
        - Box Entries (box_entries)
        - Offensive duel win rate (offdwr)
        """)

    with st.expander("**Passing Effectiveness**: Metrics assessing the team's passing accuracy and effectiveness in offensive play."):
        st.write("""
        - Total passing accuracy (pass_success_rate)
        - Total ball losses (losses)
        - Total ball recoveries (recoveries)
        - Total duels attempted (tot_duels)
        - Forward passes success rate (succ_for_passes_rate)
        - Back passes success rate (succ_back_passes_rate)
        - Lateral passes success rate (succ_lat_passes_rate)
        - Long passes success rate (succ_long_passes_rate)
        - Final third passes success rate (succ_fin_third_passes_rate)
        - Progressive passes success rate (succ_prog_passes_rate)
        - Smart passes success rate (succ_smart_passes_rate)
        - Throw-ins success rate (succ_throw_in_rate)
        - Average passes per possession (mean_pass_per_poss)
        - Average pass length (mean_pass_len)
        - Passes per defensive action (ppda)
        """)

    st.write("""
    ### Defensive Metrics Comparison
    The defensive metrics section analyzes the team's defensive strength. It includes the following categories:
    """)

    with st.expander("**Defensive Solidity**: Metrics related to the team's ability to defend against attacks."):
        st.write("""
        - Shots on target rate conceded (sotr_against)
        - Defensive duel win rate (defdwr)
        - Aerial duels win rate (airdwr)
        - Slide tackle success rate (slide_tackled_succ_rate)
        """)

    with st.expander("**Ball Recovery**: Metrics assessing how well the team recovers possession from opponents."):
        st.write("""
        - Total recoveries of the ball (recoveries)
        - Total recoveries of the ball in their own third (recovery_low)
        - Total recoveries of the ball in the middle third (recovery_med)
        - Total recoveries of the ball in the final third (recovery_high)
        """)

    with st.expander("**Pressing Efficiency**: Metrics related to the team’s pressing game and its effectiveness in disrupting opponent possession."):
        st.write("""
        - Pressing intensity (ppda)
        - Possession (possession)
        - Average passes per possession (mean_pass_per_poss)
        """)

    st.write("""
    ### Other Metrics Comparison
    The "Other" section includes additional performance indicators that don't fit neatly into offensive or defensive categories. It includes the following categories:
    """)

    with st.expander("**Possession and Passing**: Metrics related to possession control and the effectiveness of passing."):
        st.write("""
        - Passing accuracy (pass_success_rate)
        - Match tempo (match_tempo)
        - Long pass rate (long_pass_rate)
        - Average passes per possession (mean_pass_per_poss)
        - Average pass length (mean_pass_len)
        - Average shot distance (mean_shot_dist)
        """)

    with st.expander("**Discipline**: Metrics concerning player behavior, such as fouls and card accumulation."):
        st.write("""
        - Total number of fouls (fouls)
        - Total number of yellow cards (yellows)
        - Total number of red cards (reds)
        """)

    st.image('soccer_roster.png', caption='2024 NorthwesternMen\'s Soccer Roster', use_container_width=True)

