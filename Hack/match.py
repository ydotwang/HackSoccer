# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import openai
import re
import copy

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in the .env file.")

# Function to clean team names
def clean_team_name(name):
    """
    Cleans and normalizes team names by removing parenthetical expressions,
    common suffixes, extra spaces, and converting to lowercase.
    """
    # Remove parenthetical expressions like (P), (E), etc.
    name = re.sub(r'\s*\(.*?\)\s*', '', name)
    # Remove common suffixes
    name = re.sub(r'\b(University|College|State|Institute|Academy)\b', '', name, flags=re.IGNORECASE)
    # Remove extra spaces and convert to lowercase
    name = re.sub(r'\s+', ' ', name).strip().lower()
    return name

# Function to infer win/loss from match details (binary classification)
def infer_win_loss(row):
    """
    Infers whether the team won or not based on match details.
    Returns 1 for win and 0 for loss or draw.
    """
    match_details = row['match'].strip()
    team_name = clean_team_name(row['team'].strip())

    # Extract all scores from the match details
    scores = re.findall(r'(\d+):(\d+)', match_details)
    if not scores:
        raise ValueError(f"Scores not found in match details: '{match_details}'")
    score_a_str, score_b_str = scores[-1]
    score_a = int(score_a_str)
    score_b = int(score_b_str)

    # Remove scores and any text after them to isolate team names
    match_details_no_score = re.sub(r'\s+\d+:\d+.*$', '', match_details)
    if ' - ' not in match_details_no_score:
        raise ValueError(f"Expected ' - ' delimiter in match details: '{match_details_no_score}'")
    teams = match_details_no_score.split(' - ', 1)

    # Clean and normalize team names
    team_a = clean_team_name(teams[0])
    team_b = clean_team_name(teams[1])

    # Determine which team is the subject
    if team_name == team_a:
        team_found = 'A'
    elif team_name == team_b:
        team_found = 'B'
    else:
        raise ValueError(f"Team '{team_name.title()}' not found in match details: '{match_details}'")

    # Determine the match outcome (binary classification)
    if team_found == 'A':
        return 1 if score_a > score_b else 0  # Win or Not Win
    else:
        return 1 if score_b > score_a else 0  # Win or Not Win

# Function to perform Feature Sensitivity Analysis
def feature_sensitivity_analysis(model, scaler, feature_columns, X_val_original, y_val, step_size=5, top_n=5):
    """
    Analyzes how incremental changes in individual features affect the predicted win rate.

    Parameters:
    - model: Trained sklearn model.
    - scaler: Fitted StandardScaler object.
    - feature_columns: List of feature column names.
    - X_val_original: Original (unscaled) validation features (numpy array).
    - y_val: Original validation targets (numpy array).
    - step_size: Step size for feature modifications.
    - top_n: Number of top features to display based on impact.

    Returns:
    - DataFrame summarizing the impact of each feature modification.
    """
    results_list = []
    original_win_rate = np.mean(y_val)
    print(f"Original Win Rate: {original_win_rate * 100:.2f}%")

    for feature in feature_columns:
        max_val = team_data_filtered[feature].max()
        min_val = team_data_filtered[feature].min()

        # Determine step size based on feature scale
        if team_data_filtered[feature].dtype in [float, np.float64]:
            # For float features, use a fraction of the max value
            # Here, using step of 5% of max or minimum step of 0.05
            step = max(0.05, 0.05 * max_val)
        else:
            # For integer features, use step_size
            step = step_size

        # Define the range of values to iterate over
        values = np.arange(0, max_val + step, step)
        values = np.round(values, decimals=2)  # Round for cleaner values

        win_rates = []

        for val in values:
            # Create a modified copy of X_val_original
            X_modified = copy.deepcopy(X_val_original)
            feature_idx = feature_columns.index(feature)
            X_modified[:, feature_idx] = val  # Set the feature to the current value

            # Ensure realistic feature values
            X_modified = np.clip(X_modified, a_min=0, a_max=None)

            # Scale the modified features
            X_modified_scaled = scaler.transform(X_modified)

            # Predict using the model
            y_pred_modified = model.predict(X_modified_scaled)

            # Calculate modified win rate
            modified_win_rate = np.mean(y_pred_modified)
            win_rates.append(modified_win_rate)

        # Calculate win rate change from original
        win_rate_change = (np.array(win_rates) - original_win_rate) * 100

        # Append results for this feature
        results_list.append({
            'Feature': feature,
            'Modification Values': values,
            'Win Rate Change (%)': win_rate_change
        })

    # Create DataFrame for results
    results_df = pd.DataFrame(results_list)

    # For each feature, find the maximum absolute win rate change
    results_df['Max Absolute Win Rate Change (%)'] = results_df['Win Rate Change (%)'].apply(lambda x: np.max(np.abs(x)))

    # Sort by the maximum absolute change and select top_n
    results_df_sorted = results_df.sort_values(by='Max Absolute Win Rate Change (%)', ascending=False).head(top_n)

    return results_df_sorted

# Main function
def main():
    # Load the datasets
    try:
        datasets = {
            "ACC": pd.read_csv('acc-combined-data.csv'),
            "BigTen": pd.read_csv('big-ten-combined-data.csv')
        }
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading datasets: {e}")

    # Combine relevant datasets (e.g., ACC and BigTen for team-level analysis)
    team_data = pd.concat([datasets['ACC'], datasets['BigTen']], ignore_index=True)

    # Drop NaN values for simplicity
    team_data = team_data.dropna()

    # Ensure 'date' is in datetime format
    team_data['date'] = pd.to_datetime(team_data['date'])

    # Ask the user to input the team name
    user_team_input = input("Enter the team name you want to analyze: ").strip()
    selected_team = clean_team_name(user_team_input)
    print(f"Selected Team: {selected_team.title()}")

    # Filter the dataset to include only matches involving the selected team
    team_data['clean_team'] = team_data['team'].apply(clean_team_name)
    team_data_filtered = team_data[team_data['clean_team'] == selected_team]

    # Check if any matches are found
    if team_data_filtered.empty:
        raise ValueError(f"No matches found for the team '{selected_team.title()}'. Please check the team name.")

    # Apply the function to create a 'win' column
    team_data_filtered['win'] = team_data_filtered.apply(infer_win_loss, axis=1)

    # Map 'win' to 'Outcome' for better readability and color mapping
    team_data_filtered['Outcome'] = team_data_filtered['win'].map({0: 'Loss/Draw', 1: 'Win'})

    # Verify that 'win' column contains only 0 and 1
    print('Unique values in win column:', team_data_filtered['win'].unique())

    # Define feature columns and target column (excluding 'pass_success_rate', 'posit_attacks', and 'counters')
    feature_columns = [
        "shots", "sot", "sotr", "mean_shot_dist", "counters",  # Will remove 'counters'
        "tspwsr", "crosses", "acc_cross_rate", "box_entries", "touches_in_box",
        "succ_fin_third_passes_rate", "prog_passes", "succ_prog_passes_rate",
        "goals_against", "shots_against", "sot_against", "defdwr", "airdwr",
        "recoveries", "recovery_low", "recovery_med", "recovery_high", "ppda",
        "mean_pass_per_poss", "match_tempo"
    ]
    # Now remove 'counters' from feature_columns
    feature_columns.remove('counters')  

    target_column = "win"

    # Ensure selected columns exist
    missing_cols = set(feature_columns + [target_column]) - set(team_data_filtered.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Extract features and target
    X = team_data_filtered[feature_columns].values
    y = team_data_filtered[target_column].values

    # Verify that y contains only 0 and 1
    print('Unique values in y:', np.unique(y))

    # Split data into training and validation sets
    if len(y) < 10:
        raise ValueError("Not enough data points to train a model. Please select a team with more match data.")

    X_train, X_val_original, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate original win rate
    original_win_rate = np.mean(y_val)
    print(f"Original Win Rate: {original_win_rate * 100:.2f}%")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val_original)  # Scaled validation features

    # Initialize Logistic Regression Model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = lr_model.predict(X_val)
    if hasattr(lr_model, "predict_proba"):
        y_val_pred_prob = lr_model.predict_proba(X_val)[:,1]
    else:
        y_val_pred_prob = y_val_pred  # For models without predict_proba

    # Compute evaluation metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_val_pred_prob)

    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, zero_division=0))

    # Confusion Matrix using Plotly
    cm = confusion_matrix(y_val, y_val_pred)
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Loss/Draw', 'Predicted Win'],
        y=['Actual Loss/Draw', 'Actual Win'],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='Count')
    ))
    cm_fig.update_layout(
        title='Confusion Matrix - Logistic Regression',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        font=dict(size=14)
    )
    cm_fig.show()

    # Feature Importance Analysis using Logistic Regression Coefficients with Plotly
    coef = lr_model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': coef
    }).sort_values(by='Coefficient', key=lambda x: x.abs(), ascending=False)

    print("\nFeature Coefficients:")
    print(feature_importance_df)

    # Plot feature coefficients using Plotly
    coef_fig = px.bar(
        feature_importance_df.head(10),
        x='Coefficient',
        y='Feature',
        orientation='h',
        title=f'Top 10 Feature Coefficients from Logistic Regression for {selected_team.title()}',
        labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature'},
        color='Coefficient',
        color_continuous_scale='Viridis'
    )
    coef_fig.update_layout(yaxis={'categoryorder':'total ascending'}, font=dict(size=14))
    coef_fig.show()

    # Correlation Heatmap with Plotly
    corr_matrix = team_data_filtered[feature_columns + [target_column]].corr()
    corr_fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title='Feature Correlation Matrix',
        labels=dict(x="Features", y="Features", color="Correlation")
    )
    corr_fig.update_layout(font=dict(size=14))
    corr_fig.show()

    # Performance Metrics Over Time with Plotly
    # Sort data by date
    team_data_filtered = team_data_filtered.sort_values('date')

    # Comparison of Wins and Losses with Plotly
    # Boxplot of shots in wins vs losses with Plotly
    shots_box_fig = px.box(
        team_data_filtered,
        x='Outcome',
        y='shots',
        color='Outcome',
        color_discrete_map={'Loss/Draw': 'rgba(0, 0, 255, 0.6)', 'Win': 'rgba(255, 0, 0, 0.6)'},
        title=f'Number of Shots in Wins vs Losses for {selected_team.title()}',
        labels={'shots': 'Number of Shots', 'Outcome': 'Outcome'},
        points="all"  # Show all points
    )
    shots_box_fig.update_layout(
        title_font_size=18,
        xaxis_title='Outcome',
        yaxis_title='Number of Shots',
        showlegend=False,
        font=dict(size=14)
    )
    shots_box_fig.show()

    # Scatter Plot of Key Features vs Outcome with Plotly
    key_features = ['sot', 'goals_against', 'ppda', 'pass_success_rate']  # Removed 'pass_success_rate'

    for feature in key_features:
        scatter_fig = px.scatter(
            team_data_filtered,
            x=feature,
            y='Outcome',
            color='Outcome',
            color_discrete_map={'Loss/Draw': 'rgba(0, 0, 255, 0.6)', 'Win': 'rgba(255, 0, 0, 0.6)'},
            title=f'{feature.replace("_", " ").title()} vs Outcome for {selected_team.title()}',
            labels={'Outcome': 'Outcome', feature: feature.replace('_', ' ').title()},
            hover_data=[feature]
        )
        scatter_fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
        scatter_fig.update_layout(
            title_font_size=18,
            xaxis_title=feature.replace('_', ' ').title(),
            yaxis_title='Outcome',
            legend_title_text='Outcome',
            font=dict(size=14)
        )
        scatter_fig.show()

    # Histograms of Key Features by Outcome with Plotly
    for feature in key_features:
        histogram_fig = px.histogram(
            team_data_filtered,
            x=feature,
            color='Outcome',
            color_discrete_map={'Loss/Draw': 'rgba(0, 0, 255, 0.6)', 'Win': 'rgba(255, 0, 0, 0.6)'},
            title=f'Distribution of {feature.replace("_", " ").title()} by Outcome for {selected_team.title()}',
            labels={'Outcome': 'Outcome', feature: feature.replace('_', ' ').title()},
            nbins=15  # Adjust number of bins as needed
        )
        histogram_fig.update_traces(marker_line_color='black', marker_line_width=1)
        histogram_fig.update_layout(
            title_font_size=18,
            xaxis_title=feature.replace('_', ' ').title(),
            yaxis_title='Count',
            legend_title_text='Outcome',
            font=dict(size=14)
        )
        histogram_fig.show()

    # Define feature-specific modification amounts (step size considerations)
    # For this analysis, we'll vary each feature from 0 to its maximum in steps of 5 or appropriately scaled increments
    # Define a function to determine step sizes based on feature scales
    def determine_step_size(feature, max_val):
        """
        Determines the step size for feature modification based on the feature's scale.

        Parameters:
        - feature: Feature name.
        - max_val: Maximum value of the feature in the dataset.

        Returns:
        - step: Step size for modification.
        """
        if team_data_filtered[feature].dtype in [float, np.float64]:
            # For float features, use 5% of the max value or a minimum step of 0.05
            step = max(0.05, 0.05 * max_val)
        else:
            # For integer features, use step of 5 if max > 50, else step of 1
            step = 5 if max_val > 50 else 1
        return step

    # Perform Feature Sensitivity Analysis
    print("\n=== Feature Sensitivity Analysis ===\n")
    sensitivity_results = []

    for feature in feature_columns:
        max_val = team_data_filtered[feature].max()
        step = determine_step_size(feature, max_val)

        # Define the range of values to iterate over
        values = np.arange(0, max_val + step, step)
        values = np.round(values, decimals=2)  # Round for cleaner values

        win_rates = []

        for val in values:
            # Create a modified copy of X_val_original
            X_modified = copy.deepcopy(X_val_original)
            feature_idx = feature_columns.index(feature)
            X_modified[:, feature_idx] = val  # Set the feature to the current value

            # Ensure realistic feature values
            X_modified = np.clip(X_modified, a_min=0, a_max=None)

            # Scale the modified features
            X_modified_scaled = scaler.transform(X_modified)

            # Predict using the model
            y_pred_modified = lr_model.predict(X_modified_scaled)

            # Calculate modified win rate
            modified_win_rate = np.mean(y_pred_modified)
            win_rates.append(modified_win_rate)

        # Calculate win rate change from original
        win_rate_change = (np.array(win_rates) - original_win_rate) * 100

        # Append results for this feature
        sensitivity_results.append({
            'Feature': feature,
            'Modification Values': values,
            'Win Rate Change (%)': win_rate_change
        })

    # Create DataFrame for results
    sensitivity_df = pd.DataFrame(sensitivity_results)

    # For each feature, find the maximum absolute win rate change
    sensitivity_df['Max Absolute Win Rate Change (%)'] = sensitivity_df['Win Rate Change (%)'].apply(lambda x: np.max(np.abs(x)))

    # Sort by the maximum absolute change and select top_n
    top_n = 5
    sensitivity_df_sorted = sensitivity_df.sort_values(by='Max Absolute Win Rate Change (%)', ascending=False).head(top_n)

    print("\nSensitivity Analysis Results:")
    print(sensitivity_df_sorted[['Feature', 'Max Absolute Win Rate Change (%)']])

    # Visualize the Sensitivity Analysis Results using Plotly
    for idx, row in sensitivity_df_sorted.iterrows():
        feature = row['Feature']
        values = row['Modification Values']
        win_rate_change = row['Win Rate Change (%)']

        # Calculate modified win rates
        modified_win_rates = original_win_rate * 100 + win_rate_change

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Feature Value': values,
            'Win Rate Change (%)': win_rate_change,
            'Modified Win Rate (%)': modified_win_rates
        })

        # Plot with Plotly
        fig = px.line(
            plot_df,
            x='Feature Value',
            y='Modified Win Rate (%)',
            title=f'Win Rate Change vs {feature.replace("_", " ").title()} for {selected_team.title()}',
            labels={
                'Feature Value': feature.replace("_", " ").title(),
                'Modified Win Rate (%)': 'Modified Win Rate (%)'
            },
            markers=True
        )

        fig.update_traces(line=dict(color='blue'), marker=dict(size=8))
        fig.update_layout(
            title_font_size=18,
            xaxis_title=feature.replace("_", " ").title(),
            yaxis_title='Win Rate (%)',
            font=dict(size=14)
        )

        fig.show()

    # Use OpenAI API to generate insights
    # Prepare the aggregated match data for analysis
    match_data_list = []
    for idx, row in team_data_filtered.iterrows():
        match_data = row[feature_columns + [target_column]].to_dict()
        match_data['date'] = row['date'].strftime('%Y-%m-%d')
        # Extract opponent's name
        match_info = row['match'].split(' - ')
        if len(match_info) > 1:
            if clean_team_name(match_info[0]) == selected_team:
                opponent = match_info[1].strip()
            else:
                opponent = match_info[0].strip()
        else:
            opponent = 'Unknown'
        match_data['opponent'] = opponent
        match_data_list.append(match_data)

    # Generate a single aggregated analysis for the team
    aggregated_data_summary = "Aggregated Match Data:\n" + "\n".join([str(md) for md in match_data_list])

    prompt = f"""
    As an expert soccer analyst, analyze the following aggregated match data for {selected_team.title()}. Provide actionable insights on how {selected_team.title()} can improve their performance in future matches. Highlight key strengths and weaknesses based on the data.

    {aggregated_data_summary}
    """

    def get_openai_analysis(prompt):
        """
        Sends a prompt to the OpenAI API and returns the generated analysis.
        Handles API errors gracefully.
        """
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {'role': 'system', 'content': 'You are an expert soccer analyst.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except openai.error.AuthenticationError:
            print("Authentication Error: Please check your OpenAI API key.")
            return "No analysis available due to authentication error."
        except openai.error.RateLimitError:
            print("Rate Limit Exceeded: Please wait and try again later.")
            return "No analysis available due to rate limit."
        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return "No analysis available due to an API error."

    # Generate analysis using OpenAI API
    analysis = get_openai_analysis(prompt)
    print("\nOpenAI Analysis:")
    print(analysis)

    # Generate a comprehensive report for the team
    def generate_team_report():
        """
        Generates a comprehensive report including visualizations and OpenAI analysis.
        """
        # Display the OpenAI analysis
        print("\n=== Comprehensive Analysis Report ===\n")
        print(analysis)

    # Generate the team report
    generate_team_report()

    # Optional: Save team_data_filtered with analyses to a CSV file
    # team_data_filtered.to_csv(f'{selected_team}_data_with_analyses.csv', index=False)

if __name__ == "__main__":
    main()
