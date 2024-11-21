# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import openai
import re
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Load NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

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
    # Remove parenthetical expressions like (P), (E), etc.
    name = re.sub(r'\s*\(.*?\)\s*', '', name)
    # Remove common suffixes
    name = re.sub(r'\b(University|College|State|Institute|Academy)\b', '', name, flags=re.IGNORECASE)
    # Remove extra spaces and convert to lowercase
    name = re.sub(r'\s+', ' ', name).strip().lower()
    return name

# Define a function to infer win/loss from match details (binary classification)
def infer_win_loss(row):
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
team_data['win'] = team_data.apply(infer_win_loss, axis=1)

# Verify that 'win' column contains only 0 and 1
print('Unique values in win column:', team_data['win'].unique())

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
missing_cols = set(feature_columns + [target_column]) - set(team_data.columns)
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Extract features and target
X = team_data[feature_columns].values
y = team_data[target_column].values

# Verify that y contains only 0 and 1
print('Unique values in y:', np.unique(y))

# Split data into training and validation sets
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class SoccerNet(nn.Module):
    def __init__(self, input_size):
        super(SoccerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
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
num_epochs = 50
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

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_val_pred = model(X_val_tensor)
    y_val_pred_class = (y_val_pred.cpu().numpy() > 0.5).astype(int).flatten()
    y_val_true = y_val_tensor.cpu().numpy().astype(int).flatten()

# Verify that y_val_true contains only 0 and 1
print('Unique values in y_val_true:', np.unique(y_val_true))

accuracy = accuracy_score(y_val_true, y_val_pred_class)
print(f"Validation Accuracy: {accuracy:.4f}")
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
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.show()

# Correlation Heatmap
# Calculate the correlation matrix
corr_matrix = team_data[feature_columns + [target_column]].corr()

# Generate a heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Performance Metrics Over Time
# Sort data by date
team_data = team_data.sort_values('date')

# Plotting pass success rate over time
plt.figure(figsize=(12, 6))
plt.plot(team_data['date'], team_data['pass_success_rate'], marker='o')
plt.title('Pass Success Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Pass Success Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comparison of Wins and Losses
# Boxplot of shots in wins vs losses
plt.figure(figsize=(8, 6))
sns.boxplot(x='win', y='shots', data=team_data)
plt.title('Shots in Wins vs Losses')
plt.xlabel('Win')
plt.ylabel('Number of Shots')
plt.show()

# Scatter Plot of Shots vs Goals Against
plt.figure(figsize=(8, 6))
sns.scatterplot(x='shots', y='goals_against', hue='win', data=team_data)
plt.title('Shots vs Goals Against')
plt.xlabel('Shots')
plt.ylabel('Goals Against')
plt.show()

# Use OpenAI API to generate insights
# Prepare the last match data
last_match_data = team_data.iloc[-1][feature_columns + [target_column]].to_dict()
data_summary = f"Match Data:\n{last_match_data}"

prompt = f"""
Based on the following soccer match data, provide a detailed analysis of the team's performance, highlight key strengths and weaknesses, and suggest specific areas for improvement.

{data_summary}
"""

def get_openai_analysis(prompt):
    import openai

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'You are an expert soccer analyst.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# Generate analysis for the last match
analysis = get_openai_analysis(prompt)
print("OpenAI Analysis:")
print(analysis)

# Add a column to store analyses (Note: Generating analyses for all matches can be costly)
team_data['analysis'] = ''

# Generate analysis for each match (Use with caution due to API usage)
# Uncomment the following code to generate analyses for all matches

# for idx, row in team_data.iterrows():
#     match_data = row[feature_columns + [target_column]].to_dict()
#     data_summary = f"Match Data:\n{match_data}"

#     prompt = f"""
#     Based on the following soccer match data, provide a detailed analysis of the team's performance, highlight key strengths and weaknesses, and suggest specific areas for improvement.

#     {data_summary}
#     """

#     try:
#         analysis = get_openai_analysis(prompt)
#         team_data.at[idx, 'analysis'] = analysis
#         print(f"Analysis generated for match {idx+1}/{len(team_data)}")
#     except Exception as e:
#         print(f"Failed to generate analysis for match {idx+1}: {e}")

# Text Analysis of OpenAI Analyses
# Combine all analyses
# Note: If you have generated analyses for all matches
# all_analyses = ' '.join(team_data['analysis'])

# For demonstration, using the analysis of the last match
all_analyses = analysis

# Tokenize and remove stopwords
tokens = nltk.word_tokenize(all_analyses)
tokens = [word.lower() for word in tokens if word.isalpha()]
filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

# Get the most common words
word_freq = Counter(filtered_tokens)
common_words = word_freq.most_common(20)

# Display the most common words
print("Most Common Words in Analyses:")
for word, freq in common_words:
    print(f"{word}: {freq}")

# Generate a report for the last match
def generate_match_report(match_index):
    row = team_data.iloc[match_index]
    analysis = row['analysis'] if row['analysis'] else "No analysis available."
    date = row['date'].strftime('%Y-%m-%d')
    opponent_info = row['match'].split(' - ')
    if len(opponent_info) > 1:
        opponent = opponent_info[1].split(' ')[0]
    else:
        opponent = 'Unknown'

    # Plot key metrics
    metrics = ['shots', 'sot', 'pass_success_rate', 'goals_against']
    values = [row[metric] for metric in metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color='skyblue')
    plt.title(f"Performance Metrics vs {opponent} on {date}")
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.show()

    # Display the OpenAI analysis
    print(f"Analysis for Match vs {opponent} on {date}:")
    print(analysis)

# Generate a report for the last match
generate_match_report(-1)

# Optional: Save team_data with analyses to a CSV file
# team_data.to_csv('team_data_with_analyses.csv', index=False)
