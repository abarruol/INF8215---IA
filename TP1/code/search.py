# search.py
# ---------
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


# Matricule 2166208 : Bathylle de La Grandière
# Matricule 2161214 : Augustin Barruol


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 1 ICI
    '''

    # Initialisation de l'état de départ, de la fringe et les états visités
    currentTriplet = (problem.getStartState(), None, None)
    fringe = util.Stack() # LIFO
    fringe.push(currentTriplet)
    visitedStates = dict()
    
    while not(fringe.isEmpty()):
        currentTriplet = fringe.pop()
        
        # Si le noeud actuel est l'état final
        if problem.isGoalState(currentTriplet[0]) == True:
            solution = []
            tempTriplet = currentTriplet
            
            # On remonte les directions jusqu'à l'état initial
            while tempTriplet[1] is not None:
                solution.append(tempTriplet[1])
                parent = visitedStates.get(tempTriplet[2]) # on récupère le parent du noeud actuel
                tempTriplet = (tempTriplet[2], parent[0], parent[1]) # on met à jour le noeud etudié
            solution.reverse()
            return solution
        
        # Si le noeud actuel n'est pas l'état final
        elif currentTriplet[0] not in visitedStates:
            # On parcourt tous les voisins du noeud actuel
            allNeighbors = problem.getSuccessors(currentTriplet[0])
            for neighbor in allNeighbors:
                if neighbor[0] not in visitedStates: # Si le voisin étudié n'a pas été visité
                    fringe.push((neighbor[0], neighbor[1], currentTriplet[0])) # On l'ajoute à la fringe
            visitedStates[currentTriplet[0]] = (currentTriplet[1], currentTriplet[2]) # On ajoute le point actuel aux états visités
    
    return print('No solution')


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI
    '''
    # Même architecture que la recherche en profondeur, on utilise juste une Queue (FIFO) plutôt qu'une Stack (LIFO)
    currentTriplet = (problem.getStartState(), None, None)
    fringe = util.Queue() # FIFO
    fringe.push(currentTriplet)
    visitedStates = dict()
    
    while not(fringe.isEmpty()):
        currentTriplet = fringe.pop()
        
        if problem.isGoalState(currentTriplet[0]) == True:
            solution = []
            tempTriplet = currentTriplet
            
            while tempTriplet[1] is not None:
                solution.append(tempTriplet[1])
                parent = visitedStates.get(tempTriplet[2])
                tempTriplet = (tempTriplet[2], parent[0], parent[1])
            solution.reverse()
            return solution

        elif currentTriplet[0] not in visitedStates:
            allNeighbors = problem.getSuccessors(currentTriplet[0])
            for neighbor in allNeighbors:
                if neighbor[0] not in visitedStates:
                    fringe.push((neighbor[0], neighbor[1], currentTriplet[0]))
            visitedStates[currentTriplet[0]] = (currentTriplet[1], currentTriplet[2])
    
    return print('No solution')

def uniformCostSearch(problem):
    """Search the node of least total cost first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''
    # Même architecture que la recherche en largeur, on utilise juste une PriorityQueue
    currentTriplet = (problem.getStartState(), None, None, 0) # Le 4eme paramètre est le coût pour arriver du départ jusqu'à cet état
    fringe = util.PriorityQueue()
    fringe.push(currentTriplet, 0)
    visitedStates = dict()
    
    while not(fringe.isEmpty()):
        currentTriplet = fringe.pop()
        
        if problem.isGoalState(currentTriplet[0]) == True:
            solution = []
            tempTriplet = currentTriplet
            while tempTriplet[1] is not None:
                solution.append(tempTriplet[1])
                parent = visitedStates.get(tempTriplet[2])
                tempTriplet = (tempTriplet[2], parent[0], parent[1])
            solution.reverse()
            return solution

        elif currentTriplet[0] not in visitedStates:
            allNeighbors = problem.getSuccessors(currentTriplet[0])
            for neighbor in allNeighbors:
                if neighbor[0] not in visitedStates:
                    totalCost = neighbor[2] + currentTriplet[3] # On calcule le coût pour aller du départ jusqu'à ce point
                    fringe.push((neighbor[0], neighbor[1], currentTriplet[0], totalCost), totalCost) # On pushe dans la PriorityQueue en mettant comme valeur de priorité le coût pour arriver jusqu'à l'état
            visitedStates[currentTriplet[0]] = (currentTriplet[1], currentTriplet[2])

    
    return print('No solution')

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 4 ICI
    '''

    currentTriplet = (problem.getStartState(), None, None, 0)
    fringe = util.PriorityQueue()
    fringe.push(currentTriplet, 0 + heuristic(currentTriplet[0], problem))
    
    visitedStates = dict()
    
    while not(fringe.isEmpty()):
        currentTriplet = fringe.pop()
        if problem.isGoalState(currentTriplet[0]) == True:
            solution = []
            tempTriplet = currentTriplet
            while tempTriplet[1] is not None:
                solution.append(tempTriplet[1])
                parent = visitedStates.get(tempTriplet[2])
                tempTriplet = (tempTriplet[2], parent[0], parent[1])
            solution.reverse()
            return solution

        elif currentTriplet[0] not in visitedStates:
            allNeighbors = problem.getSuccessors(currentTriplet[0])
            for neighbor in allNeighbors:
                if neighbor[0] not in visitedStates:
                    totalCostPath = neighbor[2] + currentTriplet[3] # coût pour aller du départ jusqu'à ce point
                    totalCost = totalCostPath + heuristic(neighbor[0], problem) # coût pour aller du départ jusqu'à ce point + valeur de l'heuristique associée à ce point = valeur de priorité dans la PriorityQueue
                    fringe.push((neighbor[0], neighbor[1], currentTriplet[0], totalCostPath), totalCost)
            visitedStates[currentTriplet[0]] = (currentTriplet[1], currentTriplet[2])
    
    return print('No solution')


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
