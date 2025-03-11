import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the data
@st.cache_data
def load_data():
    file_path = 'C:/Users/Bibhuti Singha/OneDrive/Desktop/proj/international_matches.csv'
    return pd.read_csv(file_path)

df = load_data()

# Process the data
teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))

# Function to get team stats
def get_team_stats(team_name):
    home_games = df[df['home_team'] == team_name]
    away_games = df[df['away_team'] == team_name]

    total_games = len(home_games) + len(away_games)
    total_wins = (home_games[home_games['home_team_result'] == 'Win'].shape[0] + 
                  away_games[away_games['home_team_result'] == 'Lose'].shape[0])
    total_losses = (home_games[home_games['home_team_result'] == 'Lose'].shape[0] +
                    away_games[away_games['home_team_result'] == 'Win'].shape[0])
    total_draws = (home_games[home_games['home_team_result'] == 'Draw'].shape[0] + 
                   away_games[away_games['home_team_result'] == 'Draw'].shape[0])

    home_wins = home_games[home_games['home_team_result'] == 'Win'].shape[0]
    home_losses = home_games[home_games['home_team_result'] == 'Lose'].shape[0]
    home_draws = home_games[home_games['home_team_result'] == 'Draw'].shape[0]

    away_wins = away_games[away_games['home_team_result'] == 'Lose'].shape[0]
    away_losses = away_games[away_games['home_team_result'] == 'Win'].shape[0]
    away_draws = away_games[away_games['home_team_result'] == 'Draw'].shape[0]

    midfield_score = np.round((df[df['home_team'] == team_name]['home_team_mean_midfield_score'].mean() + 
                               df[df['away_team'] == team_name]['away_team_mean_midfield_score'].mean()) / 2, 2)
    defense_score = np.round((df[df['home_team'] == team_name]['home_team_mean_defense_score'].mean() + 
                              df[df['away_team'] == team_name]['away_team_mean_defense_score'].mean()) / 2, 2)
    offense_score = np.round((df[df['home_team'] == team_name]['home_team_mean_offense_score'].mean() + 
                               df[df['away_team'] == team_name]['away_team_mean_offense_score'].mean()) / 2, 2)

    current_rank = df[df['home_team'] == team_name]['home_team_fifa_rank'].mean()
    rank_points = df[df['home_team'] == team_name]['home_team_total_fifa_points'].mean()

    stats = {
        'Total Games': total_games,
        'Total Wins': total_wins,
        'Total Losses': total_losses,
        'Total Draws': total_draws,
        'Home Wins': home_wins,
        'Home Losses': home_losses,
        'Home Draws': home_draws,
        'Away Wins': away_wins,
        'Away Losses': away_losses,
        'Away Draws': away_draws,
        'Current FIFA Ranking': current_rank,
        'Midfield Score': midfield_score,
        'Defense Score': defense_score,
        'Offense Score': offense_score,
        'Rank Points': rank_points,
        'Win %': np.round(100 * total_wins / total_games, 2),
        'Draw %': np.round(100 * total_draws / total_games, 2),
        'Loss %': np.round(100 * total_losses / total_games, 2)
    }
    return stats

# Create a logistic regression model
def create_model():
    df['rank_difference'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
    df['average_rank'] = df['home_team_fifa_rank'] + df['away_team_fifa_rank']
    df['point_difference'] = df['home_team_total_fifa_points'] - df['away_team_total_fifa_points']
    df['score_difference'] = df['home_team_score'] - df['away_team_score']
    df['is_won'] = df['score_difference'] > 0

    X = df[['average_rank', 'rank_difference', 'point_difference']]
    y = df['is_won']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(C=1e-5)
    features = PolynomialFeatures(degree=2)
    model = Pipeline([
        ('polynomial_features', features),
        ('logistic_regression', logreg)
    ])
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = create_model()

# Streamlit app
st.title('Football Team Analysis and Comparison')

# Create columns for LHS and RHS
col1, col2 = st.columns([1, 2])

with col1:
    st.sidebar.header('Select Team for Stats')
    team1 = st.sidebar.selectbox('Select Team:', teams)

    st.sidebar.header('Select Team for Comparison')
    team2 = st.sidebar.selectbox('Select Team to Compare with:', ['None'] + list(teams))

with col2:
    if team1:
        stats1 = get_team_stats(team1)
        st.write(f"### Stats for {team1}")
        st.write(pd.DataFrame(stats1.items(), columns=['Metric', 'Value']).set_index('Metric'))

        # Show graph for team1 stats
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(stats1.keys()), y=list(stats1.values()), ax=ax)
        ax.set_title(f'{team1} Stats')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    if team2 != 'None':
        stats2 = get_team_stats(team2)
        st.write(f"### Stats for {team2}")
        st.write(pd.DataFrame(stats2.items(), columns=['Metric', 'Value']).set_index('Metric'))

        # Show graph for team2 stats
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(stats2.keys()), y=list(stats2.values()), ax=ax)
        ax.set_title(f'{team2} Stats')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        if team1 and team2:
            st.write(f"### Comparison between {team1} and {team2}")

            comparison_df = pd.DataFrame({
                'Metric': list(stats1.keys()),
                f'{team1}': list(stats1.values()),
                f'{team2}': list(stats2.values())
            })

            st.write(comparison_df)

            # Display past match details
            past_matches = df[((df['home_team'] == team1) & (df['away_team'] == team2)) |
                              ((df['home_team'] == team2) & (df['away_team'] == team1))]

            if not past_matches.empty:
                st.write(f"### Past Matches between {team1} and {team2}")
                st.write(past_matches[['date', 'home_team', 'away_team', 'home_team_continent', 'away_team_continent',
                                       'home_team_score', 'away_team_score', 'home_team_result']])
            else:
                st.write(f"No past matches between {team1} and {team2}")

            # Predict future match outcome
            world_cup_rankings_home = df[['home_team', 'home_team_fifa_rank', 'home_team_total_fifa_points']].set_index('home_team').groupby('home_team').mean()
            world_cup_rankings_away = df[['away_team', 'away_team_fifa_rank', 'away_team_total_fifa_points']].set_index('away_team').groupby('away_team').mean()

            row = pd.DataFrame(np.array([[np.nan, np.nan, True]]), columns=X_test.columns)
            home_rank = world_cup_rankings_home.loc[team1, 'home_team_fifa_rank']
            home_points = world_cup_rankings_home.loc[team1, 'home_team_total_fifa_points']
            away_rank = world_cup_rankings_away.loc[team2, 'away_team_fifa_rank']
            away_points = world_cup_rankings_away.loc[team2, 'away_team_total_fifa_points']

            row['average_rank'] = (home_rank + away_rank) / 2
            row['rank_difference'] = home_rank - away_rank
            row['point_difference'] = home_points - away_points

            home_win_prob = model.predict_proba(row)[:, 1][0]
            st.write(f"### Prediction for {team1} vs {team2}")
            st.write(f"{team1} win probability: {home_win_prob * 100:.2f}%")
            st.write(f"{team2} win probability: {(1 - home_win_prob) * 100:.2f}%")
