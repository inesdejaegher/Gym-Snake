from queue import deque
import numpy as np

class Snake():

    """
    Fundamental data structure and state manager for an individual snake in the environment.
    The Snake class holds all pertinent information regarding the Snake's movement and body.
    The position of the snake is tracked using a queue that stores the positions of the body.

    Note:
    A potentially more space efficient implementation could track directional changes rather
    than tracking each location of the snake's body.

    The snake's body operates perfectly like a queue: when the snake moves forward, 
    a new coordinate is added to the front of the body (the neck), and the oldest coordinate 
    is removed from the back (the tail)

    !!!
    The action method only moves the head forward. It is completely up to the Controller class (in controller.py) 
    to check the grid, decide if the snake ate food, and either let it grow OR .popleft() the tail from the deque 
    to maintain its size. This separation of concerns keeps snake.py focused purely on snake mechanics, not game rules.
    """

    # Definition of the possible directions that the snake can go
    # Assign a number to each direction for easy calculation of directional properties
    # --> Opposite directions always have an absolute difference of 2 (UP-DOWN ==> |0-2|=2)
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, head_coord_start, length=3):
        """
        Function to build the snake as the game starts. It takes as arguments the coordinates of the 
        head of the snake and the initial length of the snake.

        Input:
        - head_coord_start = tuple, list, or ndarray denoting the starting coordinates for the snake's head
        - length = starting number of units in snake's body
        """

        self.direction = self.DOWN      # Defaults the snake to face down when it spawns
        self.head = np.asarray(head_coord_start).astype(int)        #  Stores the head's (x,y) coordinates as np array of integers
        self.head_color = np.array([127,0,255], np.uint8)     # Sets the color of the head of the snake (currently purple)
        
        # CRUCIAL COUNTER (especially for environments dealing with multi-step growth) 
        # --> counter tracks how many segments the snake STILL NEEDS TO GROW
        self.growth_pending = 0 #To control how much the snake grows

        self.body = deque()     # Initialise the empty deque to hold the body coordinates
        
        # Build the snake's initial body pointing straight UP from the head
        for i in range(length-1, 0, -1):
            self.body.append(self.head-np.asarray([0,i]).astype(int))

    def step(self, coord, direction):
        """
        Pure helper function --> takes any arbitrary coordinate and any direction and 
        calculates what the next coordinate would be.
        Function essentially decouples the math of grid movement from the state mutation of the snake. 
        It can be used to peek ahead without actually moving the snake yet.
        
        Input:
        - coord = list, tuple, or numpy array
        - direction = integer from 1-4 inclusive.
            0: up
            1: right
            2: down
            3: left
        """

        # Validation of the input (making sure it is one of the 4 options possible)
        assert direction < 4 and direction >= 0

        # [0,0] coordinate --> top-left of screen
        # moving up = decreasing the y-coordinate
        if direction == self.UP:
            return np.asarray([coord[0], coord[1]-1]).astype(int)
        
        # moving right = increasing the x-coordinate 
        elif direction == self.RIGHT:
            return np.asarray([coord[0]+1, coord[1]]).astype(int)
        
        # moving down = increasing the y-coordinate
        elif direction == self.DOWN:
            return np.asarray([coord[0], coord[1]+1]).astype(int)
        
        # moving left = decreasing the x-coordinate
        else:
            return np.asarray([coord[0]-1, coord[1]]).astype(int)

    def action(self, direction):
        """
        Primary method called by the game loop (in controller.py)
        This method sets a new head coordinate and appends the old head
        into the body queue. The Controller class handles popping the
        last piece of the body if no food is eaten on this step.

        The direction can be any integer value, but will be collapsed
        to 0, 1, 2, or 3 corresponding to up, right, down, left respectively.

        direction - integer from 0-3 inclusive.
            0: up
            1: right
            2: down
            3: left
        """

        # Ensure direction is either 0, 1, 2, or 3
        direction = (int(direction) % 4)

        # Prevention of illegal moves 
        # --> A snake can't instantly reverse course --> if difference of previous direction and current one is 2 --> illegal
        if np.abs(self.direction-direction) != 2:
            self.direction = direction

        self.body.append(self.head)     # Old head position is pushed to the end of the deque body (acting as the snske's new head)
        self.head = self.step(self.head, self.direction)    # Snake's head is officially moved into the new tile using the step function

        # Passes the new coordinate back to whatever called it
        return self.head 
