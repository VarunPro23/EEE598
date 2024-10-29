import numpy as np
import random
import matplotlib.pyplot as plt

# Define the grid world dimensions and rewards
grid = np.array([
    [-0.04, -0.04, -0.04, +1],
    [-0.04, None, -0.04, -1],
    [-0.04, -0.04, -0.04, -0.04]
])

# Actions: up, down, left, right
actions = ["up", "down", "left", "right"]
action_effects = {
    "up": (-1, 0), "down": (1, 0),
    "left": (0, -1), "right": (0, 1)
}

# Define parameters
alpha = 0.2  # learning rate
gamma = 0.8  # discount factor
epsilon = 0.2  # exploration rate
n_episodes = 1000  # number of episodes
n_rows, n_cols = grid.shape

# Initialize the Q-table with zeros
Q = np.zeros((n_rows, n_cols, len(actions)))

# Helper function to check if a state is terminal
def check_if_terminal(state):
    return state in [(0, 3), (1, 3)]

# Epsilon-greedy policy for action selection
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        row, col = state
        return actions[np.argmax(Q[row, col])]

# Get next state given current state and action
def get_next_state(state, action):
    row, col = state
    move = action_effects[action]
    next_row, next_col = row + move[0], col + move[1]

    # Stay in place if hitting walls or going out of bounds
    if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols or grid[next_row, next_col] is None:
        return state
    else:
        return (next_row, next_col)

# Update Q-values for SARSA
def sarsa_update(state, action, reward, next_state, next_action):
    row, col = state
    next_row, next_col = next_state
    a = actions.index(action)
    next_a = actions.index(next_action)
    
    Q[row, col, a] += alpha * (reward + gamma * Q[next_row, next_col, next_a] - Q[row, col, a])

# Update Q-values for Q-learning
def q_learning_update(state, action, reward, next_state):
    row, col = state
    next_row, next_col = next_state
    a = actions.index(action)
    
    Q[row, col, a] += alpha * (reward + gamma * np.max(Q[next_row, next_col]) - Q[row, col, a])

# Main loop for SARSA
def sarsa():

    episode_reward = []
    for episode in range(n_episodes):
        state = (2, 0)  # Starting position
        action = choose_action(state, epsilon)
        total_reward = 0

        while not check_if_terminal(state):
            next_state = get_next_state(state, action)
            reward = grid[next_state]
            next_action = choose_action(next_state, epsilon)

            sarsa_update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            total_reward += reward  # Accuulate reward for the episode
        
        episode_reward.append(total_reward)
    
    return episode_reward

# Main loop for Q-learning
def q_learning():

    episode_reward = []
    for episode in range(n_episodes):
        state = (2, 0)  # Starting position
        total_reward = 0

        while not check_if_terminal(state):
            action = choose_action(state, epsilon)
            next_state = get_next_state(state, action)
            reward = grid[next_state]

            q_learning_update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        episode_reward.append(total_reward)
    
    return episode_reward

# Run q-learning algorithm loop
q_learning_reward = q_learning()

# Extract the optimal policy for q-learning
optimal_policy = np.array([[actions[np.argmax(Q[row, col])] if grid[row, col] is not None else None
                            for col in range(n_cols)] for row in range(n_rows)])

print("Optimal Policy for q-learning:\n", optimal_policy)

# Run sarsa algorithm loop
sarsa_reward = sarsa()

# Extract the optimal policy for sarsa
optimal_policy = np.array([[actions[np.argmax(Q[row, col])] if grid[row, col] is not None else None
                            for col in range(n_cols)] for row in range(n_rows)])

print("\nOptimal Policy for sarsa:\n", optimal_policy)

# Plot episodic rewards for sarsa and q-learning

plt.figure(figsize=(10, 5))
plt.plot(sarsa_reward, label="SARSA", color="blue")
plt.plot(q_learning_reward, label="Q-learning", color="red")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Episodic Rewards Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot Q-values for each state-action pair
def plot_q_values(Q):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    fig.suptitle("Q-values for each State-Action pair")

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            if grid[row, col] is None:
                ax.set_facecolor('gray')
                ax.text(0.5, 0.5, "WALL", ha='center', va='center', color='white')
            elif check_if_terminal((row, col)):
                ax.text(0.5, 0.5, "TERMINAL", ha='center', va='center', color='black')
            else:
                for i, action in enumerate(actions):
                    ax.text(0.5, 0.2*i + 0.1, f"{action}: {Q[row, col, i]:.2f}", ha='center', va='center', fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# Plot the Q-values after training
plot_q_values(Q)

# Visualize the learned policy
def plot_policy(policy):
    plt.figure(figsize=(6, 6))
    for row in range(n_rows):
        for col in range(n_cols):
            if grid[row, col] is None:
                plt.text(col, n_rows - row - 1, "WALL", ha="center", va="center", color="gray")
            elif policy[row, col] is not None:
                plt.text(col, n_rows - row - 1, policy[row, col], ha="center", va="center", color="blue")
            else:
                plt.text(col, n_rows - row - 1, "TERMINAL", ha="center", va="center", color="black")
    plt.xticks(range(n_cols))
    plt.yticks(range(n_rows))
    plt.gca().invert_yaxis()
    plt.grid()
    plt.title("Learned Optimal Policy")
    plt.show()

# Function to extract the optimal policy from the Q-table
def get_optimal_policy(Q):
    policy = np.full((n_rows, n_cols), None)  # Initialize empty policy grid
    for row in range(n_rows):
        for col in range(n_cols):
            if grid[row, col] is not None:
                policy[row, col] = actions[np.argmax(Q[row, col])]
    return policy

# Get and plot the optimal policy for each algorithm
sarsa_policy = get_optimal_policy(Q)  # Assuming Q-table contains SARSA results
plot_policy(sarsa_policy)
