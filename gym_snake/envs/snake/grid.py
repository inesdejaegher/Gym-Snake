import numpy as np

class Grid():

    """
    This class contains all data related to the grid in which the game is contained.
    The information is stored as a numpy array of pixels.
    The grid is treated as a cartesian [x,y] plane in which [0,0] is located at
    the upper left most pixel and [max_x, max_y] is located at the lower right most pixel.

    Note that it is assumed spaces that can kill a snake have a non-zero value as their 0 channel.
    It is also assumed that HEAD_COLOR has a 255 value as its 0 channel.
    """

    # The game uses the pixel colors as the state itself
    # To check if the tile has food --> the game litteraly checks if the tile is the color of food
    BODY_COLOR = np.array([0,0,255], dtype=np.uint8)
    HEAD_COLOR = np.array([127, 0, 255], dtype=np.uint8)
    FOOD_COLOR = np.array([0,255,0], dtype=np.uint8)
    SPACE_COLOR = np.array([153,204,255], dtype=np.uint8)
    COLORS = np.asarray([BODY_COLOR, HEAD_COLOR, FOOD_COLOR, SPACE_COLOR])

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1):
        """
        Initialisation function

        Input:
        - grid_size = tuple, list, or ndarray specifying number of atomic units in
                      both the x and y direction
        - unit_size = integer denoting the atomic size of grid units in pixels
        """

        self.unit_size = int(unit_size) # size in pixels of each tile
        self.unit_gap = unit_gap # number of pixels used as a border/gap between tiles to form a checkerboard pattern.
        self.grid_size = np.asarray(grid_size, dtype=int) # number of playable tiles (size in terms of units)

        # Grid creation
        height = self.grid_size[1]*self.unit_size
        width = self.grid_size[0]*self.unit_size
        channels = 3
        self.grid = np.zeros((height, width, channels), dtype=np.uint8)

        self.grid[:,:,:] = self.SPACE_COLOR # immediately paint the grid the background color (blank space)

        # Keeps a running integer tally of how many empty tiles remain (initially 900). This is highly optimized
        # so the game knows exactly when a player wins (0 open spaces left) without recounting the board every frame.
        self.open_space = grid_size[0]*grid_size[1]

    def check_death(self, head_coord):
        """
        Checks the grid to see if argued head_coord has collided with a death space (i.e. snake or wall)

        head_coord - x,y integer coordinates as a tuple, list, or ndarray
        """
        return self.off_grid(head_coord) or self.snake_space(head_coord)

    def color_of(self, coord):
        """
        Translates a game coordinate (like [5, 5]) into pixel space (like [50, 50]) 
        and returns the 3-value RGB array located at that top-left pixel.

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return self.grid[int(coord[1]*self.unit_size), int(coord[0]*self.unit_size), :]

    def connect(self, coord1, coord2, color=BODY_COLOR):
        """
        Draws connection between two adjacent pieces using the specified color.
        Created to indicate the relative ordering of the snake's body.
        coord1 and coord2 must be adjacent.
        --> mathematically fills the gap between two adjacent blocks

        coord1 - x,y integer coordinates as a tuple, list, or ndarray
        coord2 - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        # Check for adjacency
        # Next to one another:
        adjacency1 = (np.abs(coord1[0]-coord2[0]) == 1 and np.abs(coord1[1]-coord2[1]) == 0)
        # Stacked on one another:
        adjacency2  = (np.abs(coord1[0]-coord2[0]) == 0 and np.abs(coord1[1]-coord2[1]) == 1)
        assert adjacency1 or adjacency2

        if adjacency1: # x values differ
            min_x, max_x = sorted([coord1[0], coord2[0]])
            min_x = min_x*self.unit_size+self.unit_size-self.unit_gap
            max_x = max_x*self.unit_size
            self.grid[coord1[1]*self.unit_size, min_x:max_x, :] = color
            self.grid[coord1[1]*self.unit_size+self.unit_size-self.unit_gap-1, min_x:max_x, :] = color
        else: # y values differ
            min_y, max_y = sorted([coord1[1], coord2[1]])
            min_y = min_y*self.unit_size+self.unit_size-self.unit_gap
            max_y = max_y*self.unit_size
            self.grid[min_y:max_y, coord1[0]*self.unit_size, :] = color
            self.grid[min_y:max_y, coord1[0]*self.unit_size+self.unit_size-self.unit_gap-1, :] = color

    def cover(self, coord, color):
        """
        Colors a single space on the grid. Use erase if creating an empty space on the grid.
        This function is used like draw but without affecting the open_space count.

        coord - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        # If the specified coordinate is out of the possible grid --> return a False
        if self.off_grid(coord):
            return False
        
        # Define the boundaries of the tile that has to be colored
        x = int(coord[0]*self.unit_size)
        end_x = x+self.unit_size-self.unit_gap
        y = int(coord[1]*self.unit_size)
        end_y = y+self.unit_size-self.unit_gap

        self.grid[y:end_y, x:end_x, :] = np.asarray(color, dtype=np.uint8)  # Actual coloring of the tile
        return True

    def draw(self, coord, color):
        """
        Colors a single space on the grid. Use erase if creating an empty space on the grid.
        Affects the open_space count.

        coord - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        if self.cover(coord, color):
            self.open_space -= 1
            return True
        else:
            return False

    def draw_snake(self, snake, head_color=HEAD_COLOR):
        """
        Renders a full Snake object from scratch.
        Paints the head, then loops throug the queue of body parts and temporarily pops them off the queue, 
        paints them, draws a connection line to the previous part and then pushes them back onto the queue.
        Finally it connects the head to the neck

        snake - Snake object
        head_color - [R,G,B] values as a tuple, list, or ndarray
        """

        self.draw(snake.head, head_color)   # Paint the head at the head's coordinates
        prev_coord = None   # Variable will be used to track the previously drawn body segments so the function knows where to draw the connecting lines between the adjacent segments

        # Iterate through the body (which is stored as a double-ended queue)
        for i in range(len(snake.body)):
            coord = snake.body.popleft()    # Remove oldest segment from the queue and save it in the variable 'coord'
            self.draw(coord, self.BODY_COLOR)   # Draw tail/body segment on the board using the standard 'BODY_COLOR'

            if prev_coord is not None:  # If the pervious piece has a value (meaning that this isn't the first tail piece drawn) --> use the connect() method
                self.connect(prev_coord, coord, self.BODY_COLOR)

            snake.body.append(coord)    # Add segment back onto the right side of the queue
            # By the time the for loop completes its final iteration, every piece of the snake has been popped 
            # off the left and pushed back onto the right. The queue has done a full rotation and ends up in its 
            # exact original order
            prev_coord = coord

        # Once loop finishes --> draw connection line between neck of snake and head    
        self.connect(prev_coord, snake.head, self.BODY_COLOR)

    def erase(self, coord):
        """
        Colors an eintire grid tile with SPACE_COLOR to erase potential
        connection lines and it updates the game's internal tracker to mark
        the specific tile as available.

        coord - (x,y) as tuple, list, or ndarray
        """
        # Check if tile exists on the grid
        if self.off_grid(coord):
            return False
        
        # Add to tracker that there is a new open space in the grid
        self.open_space += 1

        # Define boundaries of the grid
        x = int(coord[0]*self.unit_size)
        end_x = x+self.unit_size
        y = int(coord[1]*self.unit_size)
        end_y = y+self.unit_size

        # Color the grid tile to its default background color
        self.grid[y:end_y, x:end_x, :] = self.SPACE_COLOR
        return True

    def erase_connections(self, coord):
        """
        Function selectively paints over only the borders (the gaps) on the bottom and right sides of a specific coordinate, 
        reverting them to the background color (SPACE_COLOR).

        coord - (x,y) as tuple, list, or ndarray
        """

        # Check if tile exists on the grid
        if self.off_grid(coord):
            return False
        
        # Erase Horizontal Row Below Coord
        x = int(coord[0]*self.unit_size)
        end_x = x+self.unit_size
        y = int(coord[1]*self.unit_size)+self.unit_size-self.unit_gap
        end_y = y+self.unit_gap
        self.grid[y:end_y, x:end_x, :] = self.SPACE_COLOR

        # Erase the Vertical Column to Right of Coord
        x = int(coord[0]*self.unit_size)+self.unit_size-self.unit_gap
        end_x = x+self.unit_gap
        y = int(coord[1]*self.unit_size)
        end_y = y+self.unit_size
        self.grid[y:end_y, x:end_x, :] = self.SPACE_COLOR

        return True

    def erase_snake_body(self, snake):
        """
        Function used to systematically wipe a snake's body from the grid.
        --> Typicall invoked when a snake dies

        snake - Snake object
        """

        # Iterate over all the segments in the snake's body
        for i in range(len(snake.body)):
            self.erase(snake.body.popleft())    # Removes the oldest segment of the queue and passes that coordinate into the erase() method

    def food_space(self, coord):
        """
        Checks if argued coord is snake food

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return np.array_equal(self.color_of(coord), self.FOOD_COLOR)

    def place_food(self, coord):
        """
        Draws a food at the coord. Ensures the same placement for
        each food at the beginning of a new episode. This is useful for
        experimentation with curiosity driven behaviors.

        num - the integer denoting the 
        """
        if self.open_space < 1 or not np.array_equal(self.color_of(coord), self.SPACE_COLOR):
            return False
        self.draw(coord, self.FOOD_COLOR)
        return True

    def new_food(self):
        """
        Draws a food on a random, open unit of the grid.
        Returns true if space left. Otherwise returns false.
        """

        # Check if there is still open space left to draw a food block
        if self.open_space < 1:
            return False
        coord_not_found = True
        while(coord_not_found):
            # Generate a random coordinate (bounded by grid size)
            coord = (np.random.randint(0,self.grid_size[0]), np.random.randint(0,self.grid_size[1]))

            # Check if color of randomly generated coordinate is the same as open space
            if np.array_equal(self.color_of(coord), self.SPACE_COLOR):
                # If an open coordinate is found --> set param to False to break out while loop
                coord_not_found = False
            
        # Actually draw the food on the selected coordinate
        self.draw(coord, self.FOOD_COLOR)
        return True

    def off_grid(self, coord):
        """
        Checks if argued coord is off of the grid

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return coord[0]<0 or coord[0]>=self.grid_size[0] or coord[1]<0 or coord[1]>=self.grid_size[1]

    def snake_space(self, coord):
        """
        Checks if argued coord is occupied by a snake

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        color = self.color_of(coord)
        return np.array_equal(color, self.BODY_COLOR) or color[0] == self.HEAD_COLOR[0]
