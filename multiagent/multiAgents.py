# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        food_list = newFood.asList()

        ghost_distance = manhattanDistance(newGhostStates[0].getPosition(),newPos)
        food_distance_list = [manhattanDistance(x,newPos) for x in food_list]
        # return ghost_distance + min_food_distance
        # print(foodList)
        # newGhostStates is a list with only one element
        # The format is like:
        # Ghost: (x,y)=(12.0, 7.0), East
        result = successorGameState.getScore()

        if len(food_distance_list):
            min_food_distance = min(food_distance_list)
            result = result + 12 /min_food_distance

        if ghost_distance:
            result = result - 12 / ghost_distance
        # print("Socre is ", successorGameState.getScore())
        return result

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # This helper function aims to return the agent's action
        def helper_minimax(gameState, depth, agentIndex):

            # Judge if all the agents have taken action in this turn
            # if so, increase the depth and initialize the agentIndex
            if agentIndex == gameState.getNumAgents():
                depth = depth + 1
                agentIndex = 0

            # If game end, or reach the limited depth
            # return the action decided from the evaluationFunction
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                #print(self.evaluationFunction(gameState))
                return self.evaluationFunction(gameState)
            # If this agent is *Pacman*, set the success_val to be *-inf*
            # If a ghost, set the success_val to be *+inf*
            if agentIndex == 0:
                decision = [-9999, ""]
            else:
                decision = [9999, ""]

            # All the legal actions for current agent
            actionList = gameState.getLegalActions(agentIndex)

            for action in actionList:
                successor = gameState.generateSuccessor(agentIndex, action)
                # print("successor is :")
                # print(successor)
                successor_decision = helper_minimax(successor, depth, agentIndex + 1)
                # print("succ-decision is ", successor_decision)

                if type(successor_decision) is float:
                    succ_value = successor_decision
                else:
                    succ_value = successor_decision[0]


                # Pacman takes the action that is the best for the score
                if agentIndex == 0 and succ_value > decision[0]:
                    decision = [succ_value, action]

                # Ghost takes the action that is the worst for the score
                if agentIndex != 0 and succ_value < decision[0]:
                    decision = [succ_value, action]
            # print(decision)
            return decision

        return helper_minimax(gameState, 0, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -9999
        beta = 9999
        def helper_alphabeta(gameState, depth, agentIndex, alpha, beta):

            # Judge if all the agents have taken action in this tern
            # if so, increase the depth and initialize the agentIndex
            if agentIndex == gameState.getNumAgents():
                depth = depth + 1
                agentIndex = 0

            # If game end, or reach the limited depth
            # return the action decided from the evaluationFunction
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                #print(self.evaluationFunction(gameState))
                return self.evaluationFunction(gameState)
            # If this agent is *Pacman*, set the success_val to be *-inf*
            # If a ghost, set the success_val to be *+inf*
            if agentIndex == 0:
                decision = [-9999, ""]
            else:
                decision = [9999, ""]

            # All the legal actions for current agent
            actionList = gameState.getLegalActions(agentIndex)

            for action in actionList:
                successor = gameState.generateSuccessor(agentIndex, action)
                # print("successor is :")
                # print(successor)
                successor_decision = helper_alphabeta(successor, depth, agentIndex + 1, alpha, beta)
                # print("succ-decision is ", successor_decision)

                if type(successor_decision) is float:
                    succ_value = successor_decision
                else:
                    succ_value = successor_decision[0]
                # Pacman takes the action that is the best for the score
                if agentIndex == 0:
                    if succ_value > decision[0]:
                        decision = [succ_value, action]
                    if succ_value > beta:
                        return decision
                    alpha = max(succ_value, alpha)

                # Ghost takes the action that is the worst for the score
                if agentIndex != 0:
                    if succ_value < decision[0]:
                        decision = [succ_value, action]
                    if succ_value < alpha:
                        return decision
                    beta = min(succ_value, beta)
            # print(decision)
            return decision

        return helper_alphabeta(gameState, 0, 0, alpha,beta)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # This helper function aims to return the agent's action
        def helper_epx(gameState, depth, agentIndex):

            # Judge if all the agents have taken action in this tern
            # if so, increase the depth and initialize the agentIndex
            if agentIndex == gameState.getNumAgents():
                depth = depth + 1
                agentIndex = 0

            # If game end, or reach the limited depth
            # return the action decided from the evaluationFunction
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                #print(self.evaluationFunction(gameState))
                return self.evaluationFunction(gameState)
            # If this agent is *Pacman*, set the success_val to be *-inf*
            # If a ghost, set the success_val to be *+inf*
            if agentIndex == 0:
                decision = [-9999, ""]
            else:
                decision = [0, ""]

            # All the legal actions for current agent
            actionList = gameState.getLegalActions(agentIndex)

            for action in actionList:
                successor = gameState.generateSuccessor(agentIndex, action)
                # print("successor is :")
                # print(successor)
                successor_decision = helper_epx(successor, depth, agentIndex + 1)
                # print("succ-decision is ", successor_decision)

                if type(successor_decision) is float:
                    succ_value = successor_decision
                else:
                    succ_value = successor_decision[0]
                # Pacman takes the action that is the best for the score
                if agentIndex == 0 and succ_value > decision[0]:
                    decision = [succ_value, action]

                # Ghost takes the action that is the average of all succ_values
                if agentIndex != 0:
                    decision = [succ_value / len(actionList) + decision[0], action]
            # print(decision)
            return decision

        return helper_epx(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Idea1: use reflex
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curGhostStates = currentGameState.getGhostStates()

    numFood = currentGameState.getNumFood()

    food_list = curFood.asList()

    food_distance_list = [manhattanDistance(x, curPos) for x in food_list]

    if len(food_distance_list):
        min_food_distance = min(food_distance_list)
    else:
        min_food_distance = 0

    result = currentGameState.getScore()

    ghostFeature = 0

    for ghost in currentGameState.getGhostStates():

        curGhostDist = util.manhattanDistance(ghost.getPosition(), curPos)

        if ghost.scaredTimer > curGhostDist:
            ghostFeature = ghostFeature - curGhostDist + 150

    return result - min_food_distance - 10 * numFood + ghostFeature

# Abbreviation
better = betterEvaluationFunction
