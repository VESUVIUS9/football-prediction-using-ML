import streamlit as st
import pandas as pd
import random
import numpy as np

# Function for Tournament Simulation UI
def tournament_simulation_ui():
    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        df_teams = pd.read_csv(uploaded_file)
        
        required_columns = [
            'home_team_fifa_rank', 'away_team_fifa_rank', 
            'home_team', 'away_team'
        ]
        
        for col in required_columns:
            if col not in df_teams.columns:
                st.error(f"Dataset must contain the '{col}' column")
                break
        else:
            team_options = df_teams['home_team'].unique()
            selected_teams = st.multiselect("Select 32 teams for the tournament", team_options, max_selections=32)

            n_simulations = st.number_input("Number of simulations", min_value=1, max_value=1000, value=1)

            if len(selected_teams) == 32:
                df_selected_teams = df_teams[df_teams['home_team'].isin(selected_teams)]

                df_selected_teams['home_team_win_prob'] = 1 / df_selected_teams['home_team_fifa_rank']
                df_selected_teams['away_team_win_prob'] = 1 / df_selected_teams['away_team_fifa_rank']
                
                total_prob = df_selected_teams['home_team_win_prob'] + df_selected_teams['away_team_win_prob']
                df_selected_teams['home_team_win_prob'] /= total_prob
                df_selected_teams['away_team_win_prob'] /= total_prob
                
                def run_tournament_simulation(teams, n_simulations):
                    simulation_results = []
                    
                    progress_bar = st.progress(0)
                    
                    for sim_num in range(n_simulations):
                        random.shuffle(teams)

                        groups = [teams[i:i + 4] for i in range(0, 32, 4)]
                        group_stage_results = []
                        for group in groups:
                            group_results = simulate_group_stage(group, df_selected_teams)
                            group_stage_results.append(group_results)
                        
                        knockout_teams = [team for group in group_stage_results for team in group]

                        round_of_16_winners = simulate_knockout_stage(knockout_teams, df_selected_teams)
                        quarter_finals_winners = simulate_knockout_stage(round_of_16_winners, df_selected_teams)
                        semi_finals_winners = simulate_knockout_stage(quarter_finals_winners, df_selected_teams)
                        final_winner = simulate_knockout_stage(semi_finals_winners, df_selected_teams)[0]

                        simulation_results.append(final_winner)
                        
                        progress_bar.progress((sim_num + 1) / n_simulations)
                    
                    return simulation_results

                def simulate_group_stage(group, df_selected_teams):
                    group_results = []
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            home_team, away_team = group[i], group[j]
                            home_team_prob = df_selected_teams.loc[df_selected_teams['home_team'] == home_team, 'home_team_win_prob'].values[0]
                            away_team_prob = df_selected_teams.loc[df_selected_teams['away_team'] == away_team, 'away_team_win_prob'].values[0]

                            winner = home_team if random.random() < home_team_prob else away_team
                            group_results.append(winner)
                    return group_results

                def simulate_knockout_stage(teams, df_selected_teams):
                    winners = []
                    for i in range(0, len(teams), 2):
                        team1, team2 = teams[i], teams[i + 1]
                        team1_prob = df_selected_teams.loc[df_selected_teams['home_team'] == team1, 'home_team_win_prob'].values[0]
                        team2_prob = df_selected_teams.loc[df_selected_teams['away_team'] == team2, 'away_team_win_prob'].values[0]

                        winner = team1 if random.random() < team1_prob else team2
                        winners.append(winner)
                    return winners

                simulation_results = run_tournament_simulation(selected_teams, n_simulations)

                st.write("#### Tournament Simulation Results")
                for i, result in enumerate(simulation_results, 1):
                    st.write(f"Simulation {i}: {result}")

                win_counts = pd.Series(simulation_results).value_counts()

                most_wins_team = win_counts.idxmax()
                most_wins_count = win_counts.max()

                st.write(f"#### Team with Most Wins")
                st.write(f"The team that won the most simulations win is {most_wins_team} : {most_wins_count} wins.")

                
                st.write("#### Updated Team Data with Win Probabilities")
                st.write(df_selected_teams[['home_team', 'away_team', 'home_team_win_prob', 'away_team_win_prob']])

            else:
                st.warning("Please select exactly 32 teams to continue.")

# Function for Match Analysis UI
def match_analysis_ui():
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

    # Create a selectbox for teams
    team_name = st.selectbox("Choose a team for match analysis", teams)

    if team_name:
        stats = get_team_stats(team_name)

        # Display stats
        st.write(f"### {team_name} - Match Statistics")
        for stat, value in stats.items():
            st.write(f"{stat}: {value}")

    # Visualizations (Example of team performance)
    st.subheader(f"{team_name} Performance Metrics")
    fig, ax = plt.subplots()
    labels = ['Wins', 'Draws', 'Losses']
    sizes = [stats['Total Wins'], stats['Total Draws'], stats['Total Losses']]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'yellow', 'red'])
    ax.axis('equal')
    st.pyplot(fig)

# Main Streamlit code
def main():
    st.title("Football Match Prediction & Analysis")

    # Add a sidebar or buttons for navigation
    menu = ["Tournament Simulation", "Match Analysis"]
    choice = st.sidebar.radio("Select Option", menu)

    if choice == "Tournament Simulation":
        tournament_simulation_ui()
    elif choice == "Match Analysis":
        match_analysis_ui()

if __name__ == "__main__":
    main()
