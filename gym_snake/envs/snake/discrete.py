import numpy as np

class Discrete():
    """
    Implementation of a custom discrete action space for the RL environment
    """
    def __init__(self, n_actions):
        # Class is initialised with a single argument
        # --> number of actions (for a standard snake game, this number is 4)
        self.dtype = np.int32   # Sets the data type for the actions to 32-bit integers
        self.n = n_actions  # Saves the total number of possible actions
        self.actions = np.arange(self.n, dtype=self.dtype)  # Generate numpy array containing the 'legal' set of moves the environment accepts
        self.shape = self.actions.shape # Saves the dimensionality of the action array

    def contains(self, argument):
        # Validation function
        # --> Takes an incoming argument (action proposed by the agent) and checks if it exists within the legal "self.actions" array
        for action in self.actions:
            if action == argument:
                return True
        return False

    def sample(self):
        """
        Random selection of action from the action space
        Important for the Epsilon-Greedy Strategy
        """
        return np.random.choice(self.n)
