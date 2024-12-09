import numpy as np
import random

# Define the payoff matrices for each player type
# Each matrix entry is the probability of winning based on the two players' choices
'''payoff_matrices = {
    1: np.array([[5, 4, 7], [6, 5, 3], [3, 7, 5]]),  # Type 1 matrix
    2: np.array([[0.4, 0.7, 0.6], [0.5, 0.5, 0.2], [0.8, 0.4, 0.5]]),  # Type 2 matrix
    3: np.array([[0.6, 0.3, 0.4], [0.7, 0.5, 0.8], [0.2, 0.6, 0.5]])   # Type 3 matrix
}
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random



class Player:
    def __init__(self, player_type):
        self.type = player_type
        self.strategy = (1, 0)
        self.prob_win = np.array([[5, 4, 7],
                                  [6, 5, 2],
                                  [3, 8, 5]], dtype= float)
        self.score = 0  # Tracks the number of games won in a single tournament

    def choose_action(self):
        # Choose an action based on strategy probabilities
        return np.random.choice([0, 1, 2],
                                p=[self.strategy[0], self.strategy[1], 1 - self.strategy[0] - self.strategy[1]])
    def test_action(self):
        counts = {i:0 for i in range(3)}

        for i in range(100):
            print(self.choose_action())
            counts[self.choose_action()] += 1

        print(counts)


class MetaGame:
    def __init__(self, num_players = 4, add_strategy = True):
        self.players = [Player(player_type=i % 4) for i in range(num_players)]
        if add_strategy:
            for i in self.players:
                if i.type % 4 == 1:
                    i.strategy = [.5, 1/3]

    def reset_scores(self):
        # Reset scores for all players between tournaments
        for player in self.players:
            player.score = 0

    def play_game(self, player1, player2):
        # Each player chooses an action based on their strategy
        action1 = player1.choose_action()
        action2 = player2.choose_action()

        # Calculate player1's probability of winning based on both actions
        player1_prob_win = player1.prob_win[action1, action2] / (
                    player1.prob_win[action1, action2] + player2.prob_win[action2, action1])

        # Determine the winner based on the calculated probability
        winning_player = player1 if random.random() < player1_prob_win else player2
        return winning_player

    def tournament(self):
        current_round = self.players
        round_num = 1

        while len(current_round) > 1:
            random.shuffle(current_round)  # Randomly shuffle players for pairing
            next_round = []

            # Play games in pairs
            for i in range(0, len(current_round), 2):
                player1 = current_round[i]
                player2 = current_round[i + 1]
                winner = self.play_game(player1, player2)
                winner.score += 1  # Increase score to track wins
                next_round.append(winner)

            # Prepare for the next round
            current_round = next_round
            round_num += 1

        # The last player remaining is the tournament champion
        champion = current_round[0]
        return champion.type

    def simulate_tournaments(self, num_simulations=100):
        cumulative_stats = {i: 0 for i in range(4)}  # Track cumulative wins for each player type

        for i in range(num_simulations):
            # Reset player scores at the start of each tournament
            self.reset_scores()

            # Run a single tournament and get the champion's type
            champion_type = self.tournament()
            cumulative_stats[champion_type] += 1  # Increment the win count for the champion's type

        # Display cumulative results after all simulations
        print(f"\nCumulative Stats Over {num_simulations} Tournaments:")
        for player_type, wins in cumulative_stats.items():
            print(f"Player Type {player_type}: {wins} tournament wins")


    def simulate_games(self, num_simulations = 10000):
        results = {i: 0 for i in range(4)}
        p1 = Player(player_type= 1)
        p2 = Player(player_type= 2)
        p1.strategy = (.5, .33)
        p2.strategy = (0, 0)
        for _ in range(num_simulations):
            # player1, player2 = self.players[i], self.players[j]
            player1, player2 = p1, p2

            winner = self.play_game(player1, player2)
            results[winner.type] += 1
        print(f"Results: {results[1]}, {results[2]}")
        '''
        
        for i in range(2):
            for j in range(2):
                if i != j:
                    for _ in range(num_simulations):
                        # player1, player2 = self.players[i], self.players[j]
                        player1, player2 = p1, p2
                        counts[player1.type] += player1.choose_action()

                        winner = self.play_game(player1, player2)
                        results[winner.type] += 1
                    print(self.players[i].strategy)
        print(f"Game Wins After {num_simulations} Simulations: {results}")
        print(counts)'''

    def adjust_player_strategies_after_tournament(self):
        """
        Simulates a tournament and adjusts player payoff matrices according to their type.
        Type 0: No changes.
        Type 1: Each value in the payoff matrix is multiplied by a random number in [0.5, 2].
        Type 2: Each value in the payoff matrix is multiplied by a random number in [0.9, 1.1].
        """
        # Run the tournament

        for player in self.players:
            if player.type == 1:
                # Multiply each entry in the player's payoff matrix by a random value between 0.5 and 2
                random_multiplier = np.random.uniform(0.9, 1.11111)
                player.prob_win *= random_multiplier
                # print(f"Adjusted Player Type 1 matrix with multiplier {random_multiplier}\n{player.prob_win}\n")

            elif player.type == 2:
                # Multiply each entry in the player's payoff matrix by a random value between 0.9 and 1.1
                random_multiplier = np.random.uniform(0.8, 1.25)
                player.prob_win *= random_multiplier
                # print(f"Adjusted Player Type 2 matrix with multiplier {random_multiplier}\n{player.prob_win}\n")
            elif player.type == 3:
                # Multiply each entry in the player's payoff matrix by a random value between 0.9 and 1.1
                random_multiplier = np.random.uniform(0.5, 2)
                player.prob_win *= random_multiplier
                # print(f"Adjusted Player Type 2 matrix with multiplier {random_multiplier}\n{player.prob_win}\n")

    def simulate_tournament_over_time(self, n = 10):
        """Simulates n tournaments, adjusts player strategies after each tournament,
        and tracks the number of wins for each player type.

        Args:
            n (int): Number of tournaments to simulate.
        """
        winner_counts = [-1 for i in range(n)]  # Count wins for each player type

        for i in range(n):
            # Run a tournament and adjust player strategies after
            champion_type = self.tournament()  # Run the tournament and get the winner's type
            winner_counts[i] = champion_type  # Track the type of the tournament winner
            self.adjust_player_strategies_after_tournament()  # Adjust the players' strategies

        print(winner_counts)
        return winner_counts  # Return the winner counts if needed

    def track_winner_percentages(self, n=100):
        """Tracks the cumulative win percentages for each player type over time."""
        winner_counts = {i: 0 for i in range(4)}  # Count wins for each player type
        cumulative_percentages = {i: [] for i in range(4)}  # Track percentages over time

        for i in range(n):
            champion_type = self.tournament()
            winner_counts[champion_type] += 1

            total_tournaments = i + 1
            for player_type in winner_counts:
                percentage = (winner_counts[player_type] / total_tournaments) * 100
                cumulative_percentages[player_type].append(percentage)

            self.adjust_player_strategies_after_tournament()  # Adjust strategies

        return cumulative_percentages

    def animate_winner_percentages(self, n=100):
        """Animates the evolution of cumulative win percentages for each player type."""
        cumulative_percentages = self.track_winner_percentages(n)
        fig, ax = plt.subplots()

        x_data = list(range(n))
        y_data = {i: cumulative_percentages[i] for i in range(4)}
        colors = ['red', 'blue', 'green', 'orange']

        lines = []
        for i in range(4):
            (line,) = ax.plot([], [], label=f'Player Type {i}', color=colors[i], lw=2)
            lines.append(line)

        ax.set_xlim(0, n)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Number of Tournaments')
        ax.set_ylabel('Cumulative Win Percentage (%)')
        ax.set_title('Cumulative Win Percentages Over Time')
        ax.legend()

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i, line in enumerate(lines):
                line.set_data(x_data[:frame], y_data[i][:frame])
            return lines

        ani = animation.FuncAnimation(fig, update, frames=n, init_func=init, blit=True, repeat=False)
        plt.show()

    def animate_winner_distribution(self, n=100):
        """Animates the evolution of total wins for each player type as a bar chart."""
        winner_counts = {i: 0 for i in range(4)}  # Count wins for each player type
        fig, ax = plt.subplots()

        player_types = list(winner_counts.keys())
        bar_colors = ['red', 'blue', 'green', 'orange']

        bars = plt.bar(player_types, winner_counts.values(), color=bar_colors)
        ax.set_ylim(0, n)
        ax.set_xlabel('Player Type')
        ax.set_ylabel('Total Wins')
        ax.set_title('Tournament Wins by Player Type')

        def update(frame):
            champion_type = self.tournament()
            winner_counts[champion_type] += 1
            self.adjust_player_strategies_after_tournament()  # Adjust strategies

            for i, bar in enumerate(bars):
                bar.set_height(winner_counts[i])

            return bars

        ani = animation.FuncAnimation(fig, update, frames=n, repeat=False, blit=False)
        plt.show()





# Run 100 simulated tournaments
game = MetaGame(add_strategy= False, num_players= 8)
#game.animate_winner_percentages(n=100)
game.animate_winner_percentages(n = 100)


