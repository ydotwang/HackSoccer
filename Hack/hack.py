# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import openai
import re
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

# Load the datasets
datasets = {
    "ACC": pd.read_csv('acc-combined-data.csv'),
    "BigTen": pd.read_csv('big-ten-combined-data.csv')
}

# Combine relevant datasets (e.g., ACC and BigTen for team-level analysis)
team_data = pd.concat([datasets['ACC'], datasets['BigTen']], ignore_index=True)

# Drop NaN values for simplicity
team_data = team_data.dropna()

# Ensure 'date' is in datetime format
team_data['date'] = pd.to_datetime(team_data['date'])

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

# Define a function to infer win/loss from match details (binary classification)
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

    # Check if the team from the row matches team_a or team_b
    if team_name == team_a or team_name in team_a or team_a in team_name:
        team_found = 'A'
    elif team_name == team_b or team_name in team_b or team_b in team_name:
        team_found = 'B'
    else:
        raise ValueError(f"Team '{team_name}' not found in match details: '{match_details}'")

    # Determine the match outcome (binary classification)
    if team_found == 'A':
        return 1 if score_a > score_b else 0  # Win or Not Win
    else:
        return 1 if score_b > score_a else 0  # Win or Not Win

# Apply the function to create a 'win' column
team_data_filtered['win'] = team_data_filtered.apply(infer_win_loss, axis=1)

# Map 'win' to 'Outcome' for better readability and color mapping
team_data_filtered['Outcome'] = team_data_filtered['win'].map({0: 'Loss/Draw', 1: 'Win'})

# Verify that 'win' column contains only 0 and 1
print('Unique values in win column:', team_data_filtered['win'].unique())

# Define feature columns and target column
feature_columns = [
    "shots", "sot", "sotr", "mean_shot_dist", "posit_attacks", "counters",
    "tspwsr", "crosses", "acc_cross_rate", "box_entries", "touches_in_box",
    "succ_fin_third_passes_rate", "prog_passes", "succ_prog_passes_rate",
    "goals_against", "shots_against", "sot_against", "defdwr", "airdwr",
    "recoveries", "recovery_low", "recovery_med", "recovery_high", "ppda",
    "pass_success_rate", "mean_pass_per_poss", "match_tempo"
]
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
if len(y) < 5:
    raise ValueError("Not enough data points to train a model. Please select a team with more match data.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# Define custom Dataset class
class SoccerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader
train_dataset = SoccerDataset(X_train_tensor, y_train_tensor)
val_dataset = SoccerDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define the neural network model
class SoccerNet(nn.Module):
    def __init__(self, input_size):
        super(SoccerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Instantiate the model
input_size = X_train_tensor.shape[1]
model = SoccerNet(input_size).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 30
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * X_batch.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_val_loss += loss.item() * X_batch.size(0)
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

# Plot training and validation loss with Plotly
loss_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Training Loss': train_losses,
    'Validation Loss': val_losses
})

fig = px.line(loss_df, x='Epoch', y=['Training Loss', 'Validation Loss'],
              title=f"Training and Validation Loss for {selected_team.title()}",
              labels={'value': 'Loss', 'variable': 'Loss Type'},
              markers=True,
              line_shape='linear')

fig.update_layout(title_font_size=18,
                  xaxis_title='Epoch',
                  yaxis_title='Loss',
                  legend_title_text='Loss Type',
                  font=dict(size=14))

fig.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_val_pred = model(X_val_tensor)
    y_val_pred_class = (y_val_pred.cpu().numpy() > 0.5).astype(int).flatten()
    y_val_true = y_val_tensor.cpu().numpy().astype(int).flatten()

# Verify that y_val_true contains only 0 and 1
print('Unique values in y_val_true:', np.unique(y_val_true))

accuracy = accuracy_score(y_val_true, y_val_pred_class)
print(f"Validation Accuracy for {selected_team.title()}: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_val_true, y_val_pred_class))

# Feature Importance Analysis using Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

# Plot feature importances with Plotly
fig = px.bar(feature_importance_df, x='Importance', y='Feature',
             orientation='h',
             title=f'Feature Importances from Random Forest for {selected_team.title()}',
             labels={'Importance': 'Importance', 'Feature': 'Features'},
             hover_data=['Importance'],
             color='Feature',
             color_discrete_sequence=px.colors.qualitative.Plotly)  # Using a Plotly qualitative palette

fig.update_layout(title_font_size=18,
                  xaxis_title='Importance',
                  yaxis_title='Features',
                  showlegend=False,
                  font=dict(size=14))

fig.show()

# Correlation Heatmap with Plotly
# Calculate the correlation matrix
corr_matrix = team_data_filtered[feature_columns + [target_column]].corr()

# Generate a heatmap with annotations and improved color palette
fig = px.imshow(corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title=f'Correlation Matrix of Features for {selected_team.title()}')

fig.update_layout(title_font_size=20,
                  xaxis_title='Features',
                  yaxis_title='Features',
                  font=dict(size=12))

fig.show()

# Performance Metrics Over Time with Plotly
# Sort data by date
team_data_filtered = team_data_filtered.sort_values('date')

# Plotting pass success rate over time with a color gradient
fig = px.scatter(team_data_filtered, x='date', y='pass_success_rate',
                 color='pass_success_rate',
                 color_continuous_scale='Bluered_r',  # Diverging color scale
                 title=f'Pass Success Rate Over Time for {selected_team.title()}',
                 labels={'date': 'Date', 'pass_success_rate': 'Pass Success Rate (%)'},
                 size_max=15)

fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
fig.update_layout(title_font_size=18,
                  xaxis_title='Date',
                  yaxis_title='Pass Success Rate (%)',
                  font=dict(size=14))

fig.show()

# Comparison of Wins and Losses with Plotly
# Boxplot of shots in wins vs losses with Plotly
fig = px.box(team_data_filtered, x='Outcome', y='shots',
             color='Outcome',
             color_discrete_map={'Loss/Draw': 'rgba(0, 0, 255, 0.6)', 'Win': 'rgba(255, 0, 0, 0.6)'},
             title=f'Number of Shots in Wins vs Losses for {selected_team.title()}',
             labels={'shots': 'Number of Shots', 'Outcome': 'Outcome'},
             points="all")  # Show all points

fig.update_layout(title_font_size=18,
                  xaxis_title='Outcome',
                  yaxis_title='Number of Shots',
                  showlegend=False,
                  font=dict(size=14))

fig.show()

# Scatter Plot of Key Features vs Outcome with Plotly
key_features = ['sot', 'pass_success_rate', 'goals_against', 'ppda']

for feature in key_features:
    fig = px.scatter(team_data_filtered, x=feature, y='Outcome',
                     color='Outcome',
                     color_discrete_map={'Loss/Draw': 'rgba(0, 0, 255, 0.6)', 'Win': 'rgba(255, 0, 0, 0.6)'},
                     title=f'{feature.replace("_", " ").title()} vs Outcome for {selected_team.title()}',
                     labels={'Outcome': 'Outcome', feature: feature.replace('_', ' ').title()},
                     hover_data=[feature])

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
    fig.update_layout(title_font_size=18,
                      xaxis_title=feature.replace('_', ' ').title(),
                      yaxis_title='Outcome',
                      legend_title_text='Outcome',
                      font=dict(size=14))

    fig.show()

# Histograms of Key Features by Outcome with Plotly
for feature in key_features:
    fig = px.histogram(team_data_filtered, x=feature, color='Outcome',
                       color_discrete_map={'Loss/Draw': 'rgba(0, 0, 255, 1)', 'Win': 'rgba(255, 0, 0, 1)'},
                       title=f'Distribution of {feature.replace("_", " ").title()} by Outcome for {selected_team.title()}',
                       labels={'Outcome': 'Outcome', feature: feature.replace('_', ' ').title()},
                       nbins=15
                       )  # Changed to 'overlay' for better visibility

    # Add black lines between histogram bars
    fig.update_traces(marker_line_color='black', marker_line_width=1)

    # Removed opacity from layout to ensure black lines are visible
    fig.update_layout(title_font_size=18,
                      xaxis_title=feature.replace('_', ' ').title(),
                      yaxis_title='Count',
                      legend_title_text='Outcome',
                      font=dict(size=14))

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
    print("\nComprehensive Analysis Report:")
    print(analysis)

# Generate the team report
generate_team_report()

# Optional: Save team_data_filtered with analyses to a CSV file
# team_data_filtered.to_csv(f'{selected_team}_data_with_analyses.csv', index=False)
