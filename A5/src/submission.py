from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.getDirection() gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    
    # ### START CODE HERE ###

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """
def getAction(self, gameState):
    def value(state, depth, agentIndex):
        # 1. Base cases: terminal state or depth exhausted
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        numAgents = state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        
        if not legalActions:
            return self.evaluationFunction(state)
        
        # 2. Determine next agent and check if a full ply is complete
        nextAgent = (agentIndex + 1) % numAgents
        # Depth decrements ONLY when the turn returns to Pac-Man (agent 0)
        nextDepth = depth - 1 if nextAgent == 0 else depth
        
        # 3. Recurrence branching based on Player type
        if agentIndex == 0:
            # Max Layer: Pac-Man wants to maximize score
            return max(
                value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) 
                for action in legalActions
            )
        else:
            # Min Layers: Ghosts want to minimize Pac-Man's score
            return min(
                value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) 
                for action in legalActions
            )

    # Root level: Pac-Man (agent 0) chooses the best initial action
    legalActions = gameState.getLegalActions(0)
    if not legalActions:
        return None

    # Pick the action that yields the maximum minimax value
    bestAction = max(
        legalActions,
        key=lambda action: value(gameState.generateSuccessor(0, action), self.depth, 1)
    )
    
    return bestAction
  
    # ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    pass
    # ### START CODE HERE ###

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    def value(state, depth, agentIndex, alpha, beta):
        # 1. Base cases: terminal state or depth exhausted
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        numAgents = state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        
        if not legalActions:
            return self.evaluationFunction(state)
        
        # 2. Determine next agent and check if a full ply is complete
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth
        
        # 3. Max Layer (Pac-Man, agentIndex == 0)
        if agentIndex == 0:
            v = -float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(successor, nextDepth, nextAgent, alpha, beta))
                if v > beta:
                    return v  # Beta cutoff
                alpha = max(alpha, v)
            return v
        
        # 4. Min Layers (Ghosts, agentIndex >= 1)
        else:
            v = float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, value(successor, nextDepth, nextAgent, alpha, beta))
                if v < alpha:
                    return v  # Alpha cutoff
                beta = min(beta, v)
            return v

    # Root level execution for Pac-Man (agent 0)
    legalActions = gameState.getLegalActions(0)
    if not legalActions:
        return None

    bestAction = None
    bestValue = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    # Evaluate actions at the root and update alpha/beta bounds
    for action in legalActions:
        successor = gameState.generateSuccessor(0, action)
        # Next agent is ghost 1, passing initial alpha and beta
        val = value(successor, self.depth, 1, alpha, beta)
        if val > bestValue:
            bestValue = val
            bestAction = action
        if bestValue > beta:
            return bestAction
        alpha = max(alpha, bestValue)

    return bestAction

    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
   
    # ### START CODE HERE ###
    def value(state, depth, agentIndex):
        # 1. Base cases: terminal state or depth exhausted
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        numAgents = state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        
        if not legalActions:
            return self.evaluationFunction(state)
        
        # 2. Determine next agent and check if a full ply is complete
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth
        
        # 3. Max Layer (Pac-Man, agentIndex == 0)
        if agentIndex == 0:
            return max(
                value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) 
                for action in legalActions
            )
        
        # 4. Chance Layers (Ghosts, agentIndex >= 1) - Uniform random choice
        else:
            totalValue = sum(
                value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) 
                for action in legalActions
            )
            return totalValue / len(legalActions)

    # Root level execution for Pac-Man (agent 0)
    legalActions = gameState.getLegalActions(0)
    if not legalActions:
        return None

    # Choose the action that maximizes the expected utility of the successor state
    bestAction = max(
        legalActions,
        key=lambda action: value(gameState.generateSuccessor(0, action), self.depth, 1)
    )
    
    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """
 
  # ### START CODE HERE ###

  if currentGameState.isWin():
      return 999999
  if currentGameState.isLose():
      return -999999

  currentScore = currentGameState.getScore()
  pacmanPos = currentGameState.getPacmanPosition()
  foodList = currentGameState.getFood().asList()
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  capsules = currentGameState.getCapsules()

  # 1. Food proximity feature (using reciprocal distance)
  closestFoodDist = min([util.manhattanDistance(pacmanPos, food) for food in foodList]) if foodList else 0
  foodProximity = 1.0 / (closestFoodDist + 1) if closestFoodDist is not None else 0

  # 2. Ghost interaction features
  ghostScore = 0
  for i, ghost in enumerate(ghostStates):
      ghostPos = ghost.getPosition()
      if ghostPos is None:
          continue
      
      dist = util.manhattanDistance(pacmanPos, ghostPos)
      scaredTime = scaredTimes[i]

      if scaredTime > 0:
          # Scared ghost: encourage getting closer to eat them
          if dist < scaredTime:
              ghostScore += 200.0 / (dist + 1)
      else:
          # Active ghost: avoid closeness aggressively
          if dist <= 1:
              return -999999  # Heavy penalty for immediate danger/death
          else:
              ghostScore -= 2.0 / (dist + 1)

  # 3. Aggregate and weight features
  features = [
      currentScore * 200,
      foodProximity * 150,
      -len(foodList) * 40,
      -len(capsules) * 30,
      ghostScore * 50
  ]

  return sum(features)
  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction





#########FOR QUESTION 5: Implementing Expectimax with a custom evaluation function

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.
    """
    def value(state, depth, agentIndex):
        # 1. Base cases: terminal state or depth exhausted
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        numAgents = state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)
        
        if not legalActions:
            return self.evaluationFunction(state)
        
        # 2. Determine next agent and check if a full ply is complete
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth
        
        # 3. Recurrence branching based on Player type
        if agentIndex == 0:
            # Max Layer: Pac-Man wants to maximize score
            return max(
                value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) 
                for action in legalActions
            )
        else:
            # Min Layers: Ghosts want to minimize Pac-Man's score
            return min(
                value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) 
                for action in legalActions
            )

    # Root level: Pac-Man (agent 0) chooses the best initial action
    legalActions = gameState.getLegalActions(0)
    if not legalActions:
        return None

    # Pick the action that yields the maximum minimax value
    bestAction = max(
        legalActions,
        key=lambda action: value(gameState.generateSuccessor(0, action), self.depth, 1)
    )
    
    return bestAction