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
import numpy as np
import random


class Player:
    def __init__(self, player_type):
        self.type = player_type
        self.strategy = (.8, .1)
        self.prob_win = np.array([[5, 4, 7],
                                  [6, 5, 2],
                                  [3, 8, 5]])
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
    def __init__(self, add_strategy = True):
        self.players = [Player(player_type=i % 4) for i in range(128)]
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

    def simulate_games(self, num_simulations = 1000):
        results = {i: 0 for i in range(4)}
        counts = {i: 0 for i in range(4)}
        p1 = Player(player_type= 1)
        p2 = Player(player_type= 2)
        p1.strategy = (1,0)
        p2.strategy = (0,1)
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
        print(counts)

# Run 100 simulated tournaments
game = MetaGame(add_strategy= True)
game.simulate_games(1000)
'''
p = Player(player_type=1)
p.test_action()
'''