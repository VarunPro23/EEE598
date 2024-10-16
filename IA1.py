import numpy as np 

# Define grid world parameters
grid_size = (3, 4)
terminal_states = {(0, 3): 1, (1, 3): -1}
walls = [(1, 1)]
reward = -0.04  # Reward for non-terminal states
gamma = 0.99  # Discount factor
theta = 1e-6  # Threshold for convergence

# Action space and corresponding directions (up, down, left, right)
actions = ['up', 'down', 'left', 'right']
action_deltas = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}
action_prob = [0.8, 0.1, 0.1]  # Probabilities for intended, left, right

# Initialize value function and policy
V = np.zeros(grid_size)
policy = np.full(grid_size, 'up', dtype=object)

# Set value of terminal states
for state, r in terminal_states.items():
    V[state] = r

def is_valid_state(state):
    """ Check if the state is within grid boundaries and not a wall. """
    i, j = state
    return 0 <= i < grid_size[0] and 0 <= j < grid_size[1] and state not in walls

def get_next_state(state, action):
    """ Compute the next state given the current state and action. """
    i, j = state
    di, dj = action_deltas[action]
    next_state = (i + di, j + dj)
    if is_valid_state(next_state):
        return next_state
    else:
        return state  # Collision with wall or boundary, stay in the same state

def compute_value_for_state(state, policy, V, gamma):
    """ Compute the expected value of a state following the policy. """
    action = policy[state]
    value = 0
    
    # Intended action (80% probability)
    next_state = get_next_state(state, action)
    value += action_prob[0] * (reward + gamma * V[next_state])
    
    # Left of intended action (10% probability)
    left_action = {'up': 'left', 'left': 'down', 'down': 'right', 'right': 'up'}[action]
    next_state = get_next_state(state, left_action)
    value += action_prob[1] * (reward + gamma * V[next_state])
    
    # Right of intended action (10% probability)
    right_action = {'up': 'right', 'right': 'down', 'down': 'left', 'left': 'up'}[action]
    next_state = get_next_state(state, right_action)
    value += action_prob[2] * (reward + gamma * V[next_state])
    
    return value

def policy_evaluation(policy, V, gamma=0.99, theta=1e-6):
    """ Perform the policy evaluation step. """
    while True:
        delta = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                if state in terminal_states or state in walls:
                    continue  # Skip terminal and wall states
                
                v = V[state]
                V[state] = compute_value_for_state(state, policy, V, gamma)
                delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
    return V

def best_action_for_state(state, V, gamma):
    """ Find the best action for a state by comparing values of all actions. """
    best_value = float('-inf')
    best_action = None
    
    for action in actions:
        value = 0
        
        # Intended action (80% probability)
        next_state = get_next_state(state, action)
        value += action_prob[0] * (reward + gamma * V[next_state])
        
        # Left of intended action (10% probability)
        left_action = {'up': 'left', 'left': 'down', 'down': 'right', 'right': 'up'}[action]
        next_state = get_next_state(state, left_action)
        value += action_prob[1] * (reward + gamma * V[next_state])
        
        # Right of intended action (10% probability)
        right_action = {'up': 'right', 'right': 'down', 'down': 'left', 'left': 'up'}[action]
        next_state = get_next_state(state, right_action)
        value += action_prob[2] * (reward + gamma * V[next_state])
        
        if value > best_value:
            best_value = value
            best_action = action
    
    return best_action

def policy_improvement(V, gamma=0.99):
    """ Perform the policy improvement step. """
    policy_stable = True
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = (i, j)
            if state in terminal_states or state in walls:
                continue  # Skip terminal and wall states
            
            old_action = policy[state]
            policy[state] = best_action_for_state(state, V, gamma)
            if old_action != policy[state]:
                policy_stable = False
    
    return policy, policy_stable

def policy_iteration(V, policy, gamma=0.99):
    """ Perform the full policy iteration process. """
    while True:
        V = policy_evaluation(policy, V, gamma)
        policy, policy_stable = policy_improvement(V, gamma)
        if policy_stable:
            break
    return policy, V

# Run policy iteration
optimal_policy, optimal_value_function = policy_iteration(V, policy, gamma)

# Print the results
print("Optimal Policy:")
for row in optimal_policy:
    print(row)

print("\nOptimal Value Function:")
print(optimal_value_function)
