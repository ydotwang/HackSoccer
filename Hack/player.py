# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from dotenv import load_dotenv
import openai
import re
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in the .env file.")

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------
# 1. General Player Data Analysis
# -----------------------------------

def general_player_analysis():
    print("\n=== General Player Data Analysis ===\n")
    
    # Load the general-player-data.csv
    try:
        general_player_data = pd.read_csv('general-player-data.csv')
    except FileNotFoundError:
        print("Error: 'general-player-data.csv' not found. Please ensure the file is in the correct directory.")
        return
    
    # Display the first few rows and column names
    print("General Player Data Sample:")
    print(general_player_data.head())
    print("\nColumn Names:")
    print(general_player_data.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['Number', 'Name', 'Position', 'Height', 'Class', 'Season', 'Team', 'Weight']
    missing_columns = [col for col in required_columns if col not in general_player_data.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}. Please check the dataset.")
        return
    
    # Handle missing values in 'Weight' by imputing with the median per team
    # This handles cases where some teams have missing 'Weight' while others don't
    general_player_data['Weight'] = general_player_data.groupby('Team')['Weight'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # For teams where 'Weight' is still NaN after median imputation (all weights missing), fill with overall median
    overall_weight_median = general_player_data['Weight'].median()
    general_player_data['Weight'] = general_player_data['Weight'].fillna(overall_weight_median)
    
    # Function to convert height to inches using regex for robustness
    def height_to_inches(height_str):
        """
        Converts height from format like "6' 5''" or "5'10''" to total inches.
        Returns np.nan if parsing fails.
        """
        try:
            match = re.match(r"(\d+)'\s*(\d+)''", height_str)
            if match:
                feet = int(match.group(1))
                inches = int(match.group(2))
                return feet * 12 + inches
            else:
                # Attempt alternative parsing
                parts = height_str.split("'")
                feet = int(parts[0].strip())
                inches = int(re.sub(r'[^\d]', '', parts[1]))
                return feet * 12 + inches
        except:
            return np.nan
    
    # Normalize height strings (replace smart quotes if any)
    general_player_data['Height'] = general_player_data['Height'].str.replace('’', "'", regex=False).str.replace('”', "''", regex=False)
    
    # Apply the function to create a new 'Height_in_inches' column
    general_player_data['Height_in_inches'] = general_player_data['Height'].apply(height_to_inches)
    
    # Print how many heights were successfully parsed
    valid_heights = general_player_data['Height_in_inches'].notna().sum()
    total_heights = len(general_player_data)
    print(f"\nSuccessfully parsed heights: {valid_heights}/{total_heights}")
    
    # Drop rows with NaN in 'Height_in_inches'
    initial_count = len(general_player_data)
    general_player_data = general_player_data.dropna(subset=['Height_in_inches'])
    final_count = len(general_player_data)
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows due to invalid 'Height' values.")
    
    # Standardize the 'Position' column by mapping similar positions to a standard set
    position_mapping = {
        'GK': 'Goalkeeper',
        'Goalkeeper': 'Goalkeeper',
        'D': 'Defender',
        'Defender': 'Defender',
        'D/M': 'Defender/Midfielder',
        'D/MF': 'Defender/Midfielder',
        'Defender': 'Defender',
        'F': 'Forward',
        'F/MF': 'Forward/Midfielder',
        'Forward': 'Forward',
        'M': 'Midfielder',
        'M/F': 'Midfielder/Forward',
        'MF': 'Midfielder',
        'MF/F': 'Midfielder/Forward',
        'Midfielder': 'Midfielder',
        'Midfielder/Defender': 'Midfielder/Defender'
    }
    
    general_player_data['Position'] = general_player_data['Position'].map(position_mapping).fillna(general_player_data['Position'])
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Position', 'Class', 'Season', 'Team']
    
    for col in categorical_cols:
        if general_player_data[col].dtype == 'object' or col == 'Season':
            le = LabelEncoder()
            general_player_data[col] = le.fit_transform(general_player_data[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded '{col}' with classes: {le.classes_}")
        else:
            print(f"Warning: Column '{col}' is not of type 'object'. Skipping encoding.")
    
    # Display the cleaned data
    print("\nCleaned General Player Data Sample:")
    print(general_player_data.head())
    
    # Check if dataset is empty after cleaning
    if general_player_data.empty:
        print("Error: No data left after cleaning. Please check the dataset for issues.")
        return
    
    # Define feature columns and target column
    feature_columns_gp = ['Height_in_inches', 'Weight', 'Class', 'Season', 'Team']
    target_column_gp = 'Position'
    
    # Extract features and target
    X_gp = general_player_data[feature_columns_gp].values
    y_gp = general_player_data[target_column_gp].values
    
    # Verify the shape
    print(f"\nFeatures shape: {X_gp.shape}")
    print(f"Target shape: {y_gp.shape}")
    
    # Check class distribution
    class_counts = pd.Series(y_gp).value_counts()
    print("\nClass distribution before splitting:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} samples")
    
    # Remove classes with less than 2 samples
    classes_to_keep = class_counts[class_counts >= 2].index
    if len(classes_to_keep) < len(class_counts):
        print("\nRemoving classes with less than 2 samples to facilitate stratified splitting.")
        mask = pd.Series(y_gp).isin(classes_to_keep)
        X_gp = X_gp[mask]
        y_gp = y_gp[mask]
        print(f"New dataset size: {X_gp.shape[0]} samples")
    
    # Check if any class still has less than 2 samples
    class_counts = pd.Series(y_gp).value_counts()
    if (class_counts < 2).any():
        print("\nError: Some classes still have less than 2 samples after removal.")
        print("Consider combining similar classes or using a different splitting strategy.")
        return
    
    # Extract the unique labels present after cleaning
    unique_labels = np.unique(y_gp)
    position_classes_filtered = [label_encoders['Position'].classes_[label] for label in unique_labels]
    
    # Split data into training and validation sets
    try:
        X_train_gp, X_val_gp, y_train_gp, y_val_gp = train_test_split(
            X_gp, y_gp, test_size=0.2, random_state=42, stratify=y_gp
        )
    except ValueError as e:
        print(f"Error during train-test split: {e}")
        return
    
    # Standardize features
    scaler_gp = StandardScaler()
    X_train_gp = scaler_gp.fit_transform(X_train_gp)
    X_val_gp = scaler_gp.transform(X_val_gp)
    
    # Convert data to PyTorch tensors
    X_train_tensor_gp = torch.tensor(X_train_gp, dtype=torch.float32).to(device)
    y_train_tensor_gp = torch.tensor(y_train_gp, dtype=torch.long).to(device)
    X_val_tensor_gp = torch.tensor(X_val_gp, dtype=torch.float32).to(device)
    y_val_tensor_gp = torch.tensor(y_val_gp, dtype=torch.long).to(device)
    
    # Define custom Dataset class
    class PlayerDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Create DataLoader
    train_dataset_gp = PlayerDataset(X_train_tensor_gp, y_train_tensor_gp)
    val_dataset_gp = PlayerDataset(X_val_tensor_gp, y_val_tensor_gp)

    train_loader_gp = DataLoader(train_dataset_gp, batch_size=16, shuffle=True)
    val_loader_gp = DataLoader(val_dataset_gp, batch_size=16, shuffle=False)

    # Define the neural network model for multi-class classification
    class PlayerNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(PlayerNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(32, num_classes)
            # No activation here since we'll use CrossEntropyLoss which includes Softmax

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.dropout1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.dropout2(out)
            out = self.fc3(out)
            return out

    # Instantiate the model
    input_size_gp = X_train_tensor_gp.shape[1]
    num_classes_gp = len(general_player_data['Position'].unique())
    model_gp = PlayerNet(input_size_gp, num_classes_gp).to(device)

    # Define loss function and optimizer
    criterion_gp = nn.CrossEntropyLoss()
    optimizer_gp = torch.optim.Adam(model_gp.parameters(), lr=0.001)

    # Train the model
    num_epochs_gp = 50
    train_losses_gp = []
    val_losses_gp = []

    print("\nStarting Training for Player Position Prediction...\n")

    for epoch in range(num_epochs_gp):
        # Training phase
        model_gp.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader_gp:
            optimizer_gp.zero_grad()
            outputs = model_gp(X_batch)
            loss = criterion_gp(outputs, y_batch)
            loss.backward()
            optimizer_gp.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= len(train_loader_gp.dataset)
        train_losses_gp.append(epoch_train_loss)

        # Validation phase
        model_gp.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader_gp:
                outputs = model_gp(X_batch)
                loss = criterion_gp(outputs, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss /= len(val_loader_gp.dataset)
        val_losses_gp.append(epoch_val_loss)

        # Print loss every 10 epochs and first epoch
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs_gp}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Plot training and validation loss with Plotly
    loss_df_gp = pd.DataFrame({
        'Epoch': range(1, num_epochs_gp + 1),
        'Training Loss': train_losses_gp,
        'Validation Loss': val_losses_gp
    })

    fig_gp = px.line(loss_df_gp, x='Epoch', y=['Training Loss', 'Validation Loss'],
                    title="Training and Validation Loss for Player Position Prediction",
                    labels={'value': 'Loss', 'variable': 'Loss Type'},
                    markers=True,
                    line_shape='linear')

    fig_gp.update_layout(title_font_size=18,
                         xaxis_title='Epoch',
                         yaxis_title='Loss',
                         legend_title_text='Loss Type',
                         font=dict(size=14))

    fig_gp.show()

    # Evaluate the model
    model_gp.eval()
    with torch.no_grad():
        outputs = model_gp(X_val_tensor_gp)
        _, predicted_gp = torch.max(outputs, 1)
        y_val_pred_gp = predicted_gp.cpu().numpy()
        y_val_true_gp = y_val_tensor_gp.cpu().numpy()

    # Calculate accuracy
    accuracy_gp = accuracy_score(y_val_true_gp, y_val_pred_gp)
    print(f"\nValidation Accuracy for Player Position Prediction: {accuracy_gp:.4f}")

    # Classification Report
    position_classes = label_encoders['Position'].classes_
    print("\nClassification Report:")
    # Update the target_names to only include classes present in y_val_true_gp
    unique_labels = np.unique(y_val_true_gp)
    position_classes_filtered = [label_encoders['Position'].classes_[label] for label in unique_labels]
    print(classification_report(y_val_true_gp, y_val_pred_gp, target_names=position_classes_filtered))

    # Feature Importance Analysis using Random Forest
    rf_model_gp = RandomForestClassifier(random_state=42)
    rf_model_gp.fit(X_train_gp, y_train_gp)

    # Get feature importances
    importances_gp = rf_model_gp.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df_gp = pd.DataFrame({
        'Feature': feature_columns_gp,
        'Importance': importances_gp
    }).sort_values(by='Importance', ascending=True)

    # Plot feature importances with Plotly
    fig_importance_gp = px.bar(feature_importance_df_gp, x='Importance', y='Feature',
                               orientation='h',
                               title='Feature Importances from Random Forest for Player Position Prediction',
                               labels={'Importance': 'Importance', 'Feature': 'Features'},
                               hover_data=['Importance'],
                               color='Feature',
                               color_discrete_sequence=px.colors.qualitative.Plotly)

    fig_importance_gp.update_layout(title_font_size=18,
                                    xaxis_title='Importance',
                                    yaxis_title='Features',
                                    showlegend=False,
                                    font=dict(size=14))

    fig_importance_gp.show()

    # Correlation Heatmap with Plotly
    corr_matrix_gp = general_player_data[feature_columns_gp + [target_column_gp]].corr()

    fig_heatmap_gp = px.imshow(corr_matrix_gp,
                               text_auto=True,
                               aspect="auto",
                               color_continuous_scale='RdBu',
                               title='Correlation Matrix of Features for Player Position Prediction')

    fig_heatmap_gp.update_layout(title_font_size=20,
                                 xaxis_title='Features',
                                 yaxis_title='Features',
                                 font=dict(size=12))

    fig_heatmap_gp.show()

    # Use OpenAI API to generate insights
    # Prepare the aggregated player data for analysis
    aggregated_player_summary = general_player_data[feature_columns_gp + [target_column_gp]].describe().to_string()

    prompt_gp = f"""
    As an expert soccer analyst, analyze the following aggregated player data for the team. Provide actionable insights on how players can improve their performance. Highlight key strengths and weaknesses based on the data.
    
    Aggregated Player Data Summary:
    {aggregated_player_summary}
    """

    analysis_gp = get_openai_analysis(prompt_gp)
    print("\nOpenAI Analysis for General Player Data:")
    print(analysis_gp)

    # Generate a comprehensive report for general-player-data.csv
    def generate_general_player_report():
        """
        Generates a comprehensive report including visualizations and OpenAI analysis for general player data.
        """
        # Display the OpenAI analysis
        print("\nComprehensive Analysis Report for General Player Data:")
        print(analysis_gp)

    # Generate the report
    generate_general_player_report()

    # Optional: Save general_player_data with analyses to a CSV file
    # general_player_data.to_csv('general_player_data_with_analyses.csv', index=False)

# -----------------------------------
# 2. Player Combined Data Analysis
# -----------------------------------

def player_combined_analysis():
    print("\n=== Player Combined Data Analysis ===\n")
    
    # Load the player-combined-data.csv
    try:
        player_combined_data = pd.read_csv('player-combined-data.csv')
    except FileNotFoundError:
        print("Error: 'player-combined-data.csv' not found. Please ensure the file is in the correct directory.")
        return
    
    # Display the first few rows and column names
    print("Player Combined Data Sample:")
    print(player_combined_data.head())
    print("\nColumn Names:")
    print(player_combined_data.columns.tolist())
    
    # Identify duplicate columns
    duplicate_columns = player_combined_data.columns[player_combined_data.columns.duplicated()].tolist()
    print(f"\nDuplicate Columns: {duplicate_columns}")
    
    # Since 'offensive_duels' is duplicated, let's rename them
    rename_dict = {}
    for col in duplicate_columns:
        if col == 'offensive_duels':
            rename_dict[col] = 'offensive_duels_1'
        elif col == 'forward_passes':
            rename_dict[col] = 'forward_passes_1'
        elif col == 'back_passes':
            rename_dict[col] = 'back_passes_1'
        # Add more renaming rules if necessary
    player_combined_data = player_combined_data.rename(columns=rename_dict)
    
    # Handle missing values by filling numerical columns with median and categorical with mode
    numerical_cols = player_combined_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols_pc = player_combined_data.select_dtypes(include=['object']).columns.tolist()
    
    # Fill numerical columns with median
    for col in numerical_cols:
        median_value = player_combined_data[col].median()
        if pd.isna(median_value):
            print(f"Warning: Column '{col}' has all NaN values. It will be dropped.")
            player_combined_data.drop(columns=[col], inplace=True)
        else:
            player_combined_data[col] = player_combined_data[col].fillna(median_value)
    
    # Fill categorical columns with mode
    for col in categorical_cols_pc:
        mode_series = player_combined_data[col].mode()
        if not mode_series.empty:
            mode_value = mode_series[0]
            player_combined_data[col] = player_combined_data[col].fillna(mode_value)
        else:
            print(f"Warning: Column '{col}' has all NaN values. It will be dropped.")
            player_combined_data.drop(columns=[col], inplace=True)
    
    # Define a function to clean team names
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
    
    # Clean team names in 'team' column
    if 'team' not in player_combined_data.columns:
        print("Error: 'team' column not found in 'player-combined-data.csv'. Please check the column names.")
        return
    player_combined_data['clean_team'] = player_combined_data['team'].apply(clean_team_name)
    
    # Extract match outcome from 'Match' column
    if 'Match' not in player_combined_data.columns:
        print("Error: 'Match' column not found in 'player-combined-data.csv'. Please check the column names.")
        return
    
    def extract_match_outcome(match_str, team):
        """
        Extracts the match outcome for the specified team.
        Returns 'Win', 'Loss', or 'Draw'.
        """
        try:
            # Example Match format: "Team A - Team B 2:1"
            # Split the string to extract teams and scores
            match_info = match_str.split(' ')
            if len(match_info) < 3:
                return 'Unknown'
            
            teams_part = ' '.join(match_info[:-1])
            score_part = match_info[-1]
            
            teams = teams_part.split('-')
            if len(teams) != 2:
                return 'Unknown'
            
            home_team = clean_team_name(teams[0].strip())
            away_team = clean_team_name(teams[1].strip())
            
            scores = score_part.split(':')
            if len(scores) != 2:
                return 'Unknown'
            
            score_home = int(scores[0])
            score_away = int(scores[1])
            
            # Determine if the team is home or away
            if team == home_team:
                if score_home > score_away:
                    return 'Win'
                elif score_home < score_away:
                    return 'Loss'
                else:
                    return 'Draw'
            elif team == away_team:
                if score_away > score_home:
                    return 'Win'
                elif score_away < score_home:
                    return 'Loss'
                else:
                    return 'Draw'
            else:
                return 'Unknown'
        except:
            return 'Unknown'
    
    # Apply the function to create 'Match_Outcome' column
    player_combined_data['Match_Outcome'] = player_combined_data.apply(
        lambda row: extract_match_outcome(row['Match'], row['clean_team']), axis=1
    )
    
    # Verify the unique values in 'Match_Outcome'
    print("\nUnique Match Outcomes:")
    print(player_combined_data['Match_Outcome'].unique())
    
    # Drop rows with 'Unknown' outcome
    initial_count = len(player_combined_data)
    player_combined_data = player_combined_data[player_combined_data['Match_Outcome'] != 'Unknown']
    final_count = len(player_combined_data)
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with 'Unknown' match outcomes.")
    
    # Encode the 'Match_Outcome' column
    label_encoder_pc = LabelEncoder()
    player_combined_data['Match_Outcome_Encoded'] = label_encoder_pc.fit_transform(player_combined_data['Match_Outcome'])
    print("\nEncoded Match Outcomes:")
    print(player_combined_data[['Match_Outcome', 'Match_Outcome_Encoded']].drop_duplicates())
    
    # Aggregate player data per team per match
    aggregation_dict = {
        'Match_Outcome_Encoded': 'first',  # To retain the outcome per match-team
        'Minutes played': 'sum',
        'total_actions': 'sum',
        'successful_actions': 'sum',
        'goals': 'sum',
        'assists': 'sum',
        'shots': 'sum',
        'shots_on_target': 'sum',
        'xg': 'sum',
        'total_passes': 'sum',
        'total_passes_completed': 'sum',
        'long_passes': 'sum',
        'long_passes_completed': 'sum',
        'crosses': 'sum',
        'accurate_crosses': 'sum',
        'dribbles': 'sum',
        'successful_dribbles': 'sum',
        'total_duels': 'sum',
        'total_duels_won': 'sum',
        'aerial_duels': 'sum',
        'aerial_duels_won': 'sum',
        'interceptions': 'sum',
        'losses': 'sum',
        'losses_in_own_half': 'sum',
        'recoveries': 'sum',
        'recoveries_in_own_half': 'sum',
        'minute_yellow_card_received': 'sum',
        'minute_red_card_received': 'sum',
        'defensive_duels': 'sum',
        'defensive_duels_won': 'sum',
        'loose_ball_duels': 'sum',
        'loose_ball_duels_won': 'sum',
        'slide_tackles_attempted': 'sum',
        'slide_tackles_won': 'sum',
        'clearances': 'sum',
        'fouls': 'sum',
        'yellow_cards': 'sum',
        'red_cards': 'sum',
        'shot_assists': 'sum',
        'offensive_duels_1': 'sum',
        'offensive_duels.1': 'sum',  # Assuming 'offensive_duels_1' is a duplicate
        'touches_inside_box': 'sum',
        'offsides': 'sum',
        'progressive_runs_attempted': 'sum',
        'fouls_drawn': 'sum',
        'through_passes_attempted': 'sum',
        'through_passes_completed': 'sum',
        'xa': 'sum',
        'second_assists': 'sum',
        'passes_into_final_third': 'sum',
        'passes_into_final_third_completed': 'sum',
        'passes_into_box': 'sum',
        'passes_into_box_completed': 'sum',
        'received_passes': 'sum',
        'forward_passes_1': 'sum',
        'forward_passes.1': 'sum',
        'back_passes_1': 'sum',
        'back_passes.1': 'sum',
        'gk_stat_conceded_goals': 'sum',
        'gk_stat_xcg': 'sum',
        'gk_stat_shots_against': 'sum',
        'gk_stat_saves': 'sum',
        'gk_stat_reflex_saves': 'sum',
        'gk_stat_box_exits': 'sum',
        'gk_stat_passes_to_gk': 'sum',
        'gk_stat_passes_to_gk_completed': 'sum',
        'gk_stat_goal_kicks_attempted': 'sum',
        'gk_stat_short_goal_kicks': 'sum',
        'gk_stat_long_goal_kicks': 'sum'
    }
    
    # Ensure that all columns in aggregation_dict exist in the dataframe
    missing_agg_cols = set(aggregation_dict.keys()) - set(player_combined_data.columns)
    if missing_agg_cols:
        print(f"Warning: The following columns are missing and will be excluded from aggregation: {missing_agg_cols}")
        for col in missing_agg_cols:
            aggregation_dict.pop(col)
    
    # Aggregate the data
    aggregated_team_data = player_combined_data.groupby(['Match', 'team']).agg(aggregation_dict).reset_index()
    
    # Display the aggregated data
    print("\nAggregated Team Data Sample:")
    print(aggregated_team_data.head())
    
    # Define feature columns and target column
    excluded_cols_pc = ['Match', 'team', 'Match_Outcome', 'Match_Outcome_Encoded']
    feature_columns_pc = [col for col in aggregated_team_data.columns if col not in excluded_cols_pc]
    target_column_pc = 'Match_Outcome_Encoded'
    
    # Extract features and target
    X_pc = aggregated_team_data[feature_columns_pc].values
    y_pc = aggregated_team_data[target_column_pc].values
    
    # Verify the shape
    print(f"\nFeatures shape: {X_pc.shape}")
    print(f"Target shape: {y_pc.shape}")
    
    # Check class distribution
    class_counts = pd.Series(y_pc).value_counts()
    print("\nClass distribution before splitting:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} samples")
    
    # Remove classes with less than 2 samples
    classes_to_keep = class_counts[class_counts >= 2].index
    if len(classes_to_keep) < len(class_counts):
        print("\nRemoving classes with less than 2 samples to facilitate stratified splitting.")
        mask = pd.Series(y_pc).isin(classes_to_keep)
        X_pc = X_pc[mask]
        y_pc = y_pc[mask]
        print(f"New dataset size: {X_pc.shape[0]} samples")
    
    # Check if any class still has less than 2 samples
    class_counts = pd.Series(y_pc).value_counts()
    if (class_counts < 2).any():
        print("\nError: Some classes still have less than 2 samples after removal.")
        print("Consider combining similar classes or using a different splitting strategy.")
        return
    
    # Extract the unique labels present after cleaning
    unique_labels = np.unique(y_pc)
    match_outcome_classes_filtered = [label_encoder_pc.classes_[label] for label in unique_labels]
    
    # Split data into training and validation sets
    try:
        X_train_pc, X_val_pc, y_train_pc, y_val_pc = train_test_split(
            X_pc, y_pc, test_size=0.2, random_state=42, stratify=y_pc
        )
    except ValueError as e:
        print(f"Error during train-test split: {e}")
        return
    
    # Standardize features
    scaler_pc = StandardScaler()
    X_train_pc = scaler_pc.fit_transform(X_train_pc)
    X_val_pc = scaler_pc.transform(X_val_pc)
    
    # Convert data to PyTorch tensors
    X_train_tensor_pc = torch.tensor(X_train_pc, dtype=torch.float32).to(device)
    y_train_tensor_pc = torch.tensor(y_train_pc, dtype=torch.long).to(device)
    X_val_tensor_pc = torch.tensor(X_val_pc, dtype=torch.float32).to(device)
    y_val_tensor_pc = torch.tensor(y_val_pc, dtype=torch.long).to(device)
    
    # Define custom Dataset class
    class TeamMatchDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Create DataLoader
    train_dataset_pc = TeamMatchDataset(X_train_tensor_pc, y_train_tensor_pc)
    val_dataset_pc = TeamMatchDataset(X_val_tensor_pc, y_val_tensor_pc)

    train_loader_pc = DataLoader(train_dataset_pc, batch_size=16, shuffle=True)
    val_loader_pc = DataLoader(val_dataset_pc, batch_size=16, shuffle=False)

    # Define the neural network model for multi-class classification
    class TeamNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(TeamNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(64, num_classes)
            # No activation here since we'll use CrossEntropyLoss which includes Softmax

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.dropout1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.dropout2(out)
            out = self.fc3(out)
            return out

    # Instantiate the model
    input_size_pc = X_train_tensor_pc.shape[1]
    num_classes_pc = len(aggregated_team_data['Match_Outcome_Encoded'].unique())
    model_pc = TeamNet(input_size_pc, num_classes_pc).to(device)

    # Define loss function and optimizer
    criterion_pc = nn.CrossEntropyLoss()
    optimizer_pc = torch.optim.Adam(model_pc.parameters(), lr=0.001)

    # Train the model
    num_epochs_pc = 50
    train_losses_pc = []
    val_losses_pc = []

    print("\nStarting Training for Match Outcome Prediction...\n")

    for epoch in range(num_epochs_pc):
        # Training phase
        model_pc.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader_pc:
            optimizer_pc.zero_grad()
            outputs = model_pc(X_batch)
            loss = criterion_pc(outputs, y_batch)
            loss.backward()
            optimizer_pc.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= len(train_loader_pc.dataset)
        train_losses_pc.append(epoch_train_loss)

        # Validation phase
        model_pc.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader_pc:
                outputs = model_pc(X_batch)
                loss = criterion_pc(outputs, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss /= len(val_loader_pc.dataset)
        val_losses_pc.append(epoch_val_loss)

        # Print loss every 10 epochs and first epoch
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs_pc}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Plot training and validation loss with Plotly
    loss_df_pc = pd.DataFrame({
        'Epoch': range(1, num_epochs_pc + 1),
        'Training Loss': train_losses_pc,
        'Validation Loss': val_losses_pc
    })

    fig_pc = px.line(loss_df_pc, x='Epoch', y=['Training Loss', 'Validation Loss'],
                    title="Training and Validation Loss for Match Outcome Prediction",
                    labels={'value': 'Loss', 'variable': 'Loss Type'},
                    markers=True,
                    line_shape='linear')

    fig_pc.update_layout(title_font_size=18,
                         xaxis_title='Epoch',
                         yaxis_title='Loss',
                         legend_title_text='Loss Type',
                         font=dict(size=14))

    fig_pc.show()

    # Evaluate the model
    model_pc.eval()
    with torch.no_grad():
        outputs = model_pc(X_val_tensor_pc)
        _, predicted_pc = torch.max(outputs, 1)
        y_val_pred_pc = predicted_pc.cpu().numpy()
        y_val_true_pc = y_val_tensor_pc.cpu().numpy()

    # Calculate accuracy
    accuracy_pc = accuracy_score(y_val_true_pc, y_val_pred_pc)
    print(f"\nValidation Accuracy for Match Outcome Prediction: {accuracy_pc:.4f}")

    # Classification Report
    match_outcome_classes = label_encoder_pc.classes_
    print("\nClassification Report:")
    # Update the target_names to only include classes present in y_val_true_pc
    unique_labels_pc = np.unique(y_val_true_pc)
    match_outcome_classes_filtered = [label_encoder_pc.classes_[label] for label in unique_labels_pc]
    print(classification_report(y_val_true_pc, y_val_pred_pc, target_names=match_outcome_classes_filtered))

    # Feature Importance Analysis using Random Forest
    rf_model_pc = RandomForestClassifier(random_state=42)
    rf_model_pc.fit(X_train_pc, y_train_pc)

    # Get feature importances
    importances_pc = rf_model_pc.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df_pc = pd.DataFrame({
        'Feature': feature_columns_pc,
        'Importance': importances_pc
    }).sort_values(by='Importance', ascending=True)

    # Plot feature importances with Plotly
    fig_importance_pc = px.bar(feature_importance_df_pc, x='Importance', y='Feature',
                               orientation='h',
                               title='Feature Importances from Random Forest for Match Outcome Prediction',
                               labels={'Importance': 'Importance', 'Feature': 'Features'},
                               hover_data=['Importance'],
                               color='Feature',
                               color_discrete_sequence=px.colors.qualitative.Plotly)

    fig_importance_pc.update_layout(title_font_size=18,
                                    xaxis_title='Importance',
                                    yaxis_title='Features',
                                    showlegend=False,
                                    font=dict(size=14))

    fig_importance_pc.show()

    # Correlation Heatmap with Plotly
    corr_matrix_pc = aggregated_team_data[feature_columns_pc + [target_column_pc]].corr()

    fig_heatmap_pc = px.imshow(corr_matrix_pc,
                               text_auto=True,
                               aspect="auto",
                               color_continuous_scale='RdBu',
                               title='Correlation Matrix of Features for Match Outcome Prediction')

    fig_heatmap_pc.update_layout(title_font_size=20,
                                 xaxis_title='Features',
                                 yaxis_title='Features',
                                 font=dict(size=12))

    fig_heatmap_pc.show()

    # Use OpenAI API to generate insights
    # Prepare the aggregated team match data for analysis
    aggregated_match_summary = aggregated_team_data[feature_columns_pc + [target_column_pc]].describe().to_string()

    prompt_pc = f"""
    As an expert soccer analyst, analyze the following aggregated team match data. Provide actionable insights on how the team can improve their performance in future matches. Highlight key strengths and weaknesses based on the data.
    
    Aggregated Match Data Summary:
    {aggregated_match_summary}
    """

    analysis_pc = get_openai_analysis(prompt_pc)
    print("\nOpenAI Analysis for Player Combined Data:")
    print(analysis_pc)

    # Generate a comprehensive report for player-combined-data.csv
    def generate_player_combined_report():
        """
        Generates a comprehensive report including visualizations and OpenAI analysis for player combined data.
        """
        # Display the OpenAI analysis
        print("\nComprehensive Analysis Report for Player Combined Data:")
        print(analysis_pc)

    # Generate the report
    generate_player_combined_report()

    # Optional: Save aggregated_team_data with analyses to a CSV file
    # aggregated_team_data.to_csv('aggregated_team_data_with_analyses.csv', index=False)

# -----------------------------------
# 3. OpenAI Analysis Function
# -----------------------------------

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

# -----------------------------------
# 4. Main Execution Flow
# -----------------------------------

def main():
    # Perform General Player Data Analysis
    general_player_analysis()
    
    # Perform Player Combined Data Analysis
    player_combined_analysis()
    
    print("\n=== Analysis Completed ===\n")

# Execute the main function
if __name__ == "__main__":
    main()
