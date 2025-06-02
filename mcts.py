from rl_environement import *
import copy
import random
import math
import numpy as np

env = ABCEnv(
        abc_path="./abc/abc",  # compiled binary path
        truth_file="./testbenches/EPFL/adder.aig"
    )
# UCT Node for MCTS
class UCTNode:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.parent = parent
        self.score = score
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions  = list(range(len(env.action_space)))  # All legal actions available at this node

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


class UCTMCTS:
    def __init__(self, env, iterations=500, exploration_constant=1.41, rollout_depth=10):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        best_action = None
        best_value = -float('inf')
        for action, child in node.children.items():
            value = child.total_reward / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        reward = 0
        orig_reward = sim_env.score
        for _ in range(depth):

            action = random.choice(sim_env.action_space_index)
            # print("Rollout action:", action)
            _, reward, done, _ = sim_env.step(action)
            
            if done:
                break
        # print("Rollout ended with reward:", reward, "original reward:", orig_reward)
        return (reward - orig_reward)/100

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
            
    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded():
            action = self.select_child(node)
            node = node.children[action]
            sim_env.step(action)
        # TODO: Expansion: if the node has untried actions, expand one.
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state, reward, done, _ = sim_env.step(action)
            new_node = UCTNode(new_state, reward, node, action)
            node.children[action] = new_node
            node = new_node
        
        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # print("Rollout reward:", rollout_reward)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node, rollout_reward)
        # print("Backpropagated reward:", rollout_reward)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(len(env.action_space_index))
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
if __name__ == "__main__":
    env = ABCEnv(
        abc_path="./abc/abc",  # compiled binary path
        truth_file="./testbenches/EPFL/adder.aig"
    )

    print("=== Resetting environment ===")
    state = env.reset()
    print("Initial observation:", state)

    print("\n=== Taking random actions ===")
    uct_mcts = UCTMCTS(env, iterations=50, exploration_constant=1.41, rollout_depth=10)
    # print(state)
    done = False
    count = 0
    while count <100:
        root = UCTNode(state, env.score)  # Initialize the root node for MCTS
        print("Running MCTS simulation...")
        # Run multiple simulations to construct and refine the search tree
        for _ in range(uct_mcts.iterations):
            uct_mcts.run_simulation(root)
        print("MCTS simulation completed. ands gate count:", env._get_observation()["ands"])
        # Select the best action based on the visit distribution of the root's children
        best_action, visit_distribution = uct_mcts.best_action_distribution(root)
        # print("MCTS selected action:", best_action, "with visit distribution:", visit_distribution)

        state, reward, done, _ = env.step(best_action)
        count += 1
    # env.render(action=best_action)  # Display the updated game state


    env.close()
