#!/usr/bin/env python3
"""
Quoridor agent.
Copyright (C) 2013, <<<<<<<<<<< YOUR NAMES HERE >>>>>>>>>>>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""

# Code implémenté par :
# Augustin BARRUOL - 2161214
# Bathylle De La Grandière - 2166208
# Pour le cours INF8215 proposé à Polytechnique Montréal


from quoridor import *
import math

class MyAgent(Agent):

    """My Quoridor agent."""
    
    def successors(self, state, game_info):
        """
        With the information of the game and the current situation, 
        this function returns a list of all the successors that are interesting for the player.
        We can not consider all the possible next state as a search time would be way too high.
        """

        board, player = state
        opponent = 1 - player
        step, time_left = game_info
        moves = []
        successors = []
        
        # The score is given by the difference of the shortest path to the goal of each player
        # If it is positive, it means our player is "closer" to the arrival than the opponent
        try:
            score_path = board.get_score(player)
        except:
            score_path = abs(board.pawns[opponent][0] - board.goals[opponent]) - abs(board.pawns[player][0] - board.goals[player])
            
        
        
        # If the opponent has no wall and the player is closer to the finish line than him, just go to the finish line
        if board.nb_walls[opponent] == 0 and score_path > 0:
            try:
                next_position = board.get_shortest_path(player)[0]
                next_action = ('P', next_position[0], next_position[1])
                new_board = board.clone()            
                new_board = new_board.play_action(next_action, player)
                return [(next_action, (new_board, opponent))]
            except:
                moves = board.get_legal_pawn_moves(player)
                for move in moves:
                    new_board = board.clone()
                    new_board = new_board.play_action(move, player)
                    successors.append((move, (new_board, opponent)))
                return successors
        
        # The first 3 moves, just move the pawn
        # Then if the score is negative or not high enough to be confident and we still have walls to place, 
        # we will try to build walls
        # It considers the walls around each position of the shortest path of the opponent and its current position
        if step > 6 and board.nb_walls[player] > 0 and score_path < 1:
            try:
                # Find relevant wall moves on the opponent's shortest path and its current position
                opponent_path = board.get_shortest_path(opponent)
                opponent_position = board.pawns[opponent]
                opponent_path.append(opponent_position)
            except:
                opponent_path = []
                opponent_position = board.pawns[opponent]
                opponent_path.append(opponent_position)
                
            # Coordinates of walls around the point (0,0)
            wall_base = [(-1,-1), (-1, 0), (0, 0), (0, -1)]
            
            # We check for walls around every position of the opponent's shortest path and its current position
            for position in opponent_path:
                for base in wall_base:
                    wall_position = (position[0] + base[0], position[1] + base[1])
                    for orientation in ['WH', 'WV']:
                        if board.is_wall_possible_here(wall_position, orientation == 'WH'):
                            moves.append((orientation, wall_position[0], wall_position[1]))
        
        # Always consider the movement to the shortest path
        try:
            next_position = board.get_shortest_path(player)[0]
            moves.append(('P', next_position[0], next_position[1]))
        except:
            displacements = board.get_legal_pawn_moves(player)
            moves.extend(displacements)
        
        successors = []
        moves_set = set(moves)
        
        for move in moves_set:
            new_board = board.clone()
            new_board = new_board.play_action(move, player)
            try:
                # We consider only the moves that augment the score
                score_temp = new_board.get_score(player)
                if score_temp >= score_path:
                    successors.append((move, (new_board, opponent)))
            except:
                successors.append((move, (new_board, opponent)))
        
        # If there is no move that augment the score, just move to the shortest path
        # (normally, must not happen)
        if len(successors) == 0:
            try:
                next_position = board.get_shortest_path(player)[0]
            except:
                next_position = board.get_legal_pawn_moves(player)[0]
            move = ('P', next_position[0], next_position[1])
            new_board = board.clone()
            new_board = new_board.play_action(move, player)
            successors.append((move, (new_board, opponent)))

        return successors
            
            

    def isTerminal(self, state, depth, game_info):
        """
        This functions returns True either when 
            - the end of the tree is finished for a branch 
            - or when we reached a certain depth depending on the time left in the game.
        """
        board, player = state
        step, time_left = game_info
        
        # For the first 6 steps, search depth = 1
        # The first few moves aren't that important and usually very similar so no need to search deep
        if step <= 6:
            return depth >= 1 or board.is_finished()
        
        # In the first 30 seconds, search depth = 3
        # The first few moves aren't that important and usually very similar so no need to search deep
        if time_left > 270:
            return depth >= 3 or board.is_finished()
        
        # Then the next 3 minutes and 30 seconds, search depth = 4
        # Mid game is really important so we need to make good choices
        if time_left > 60 and time_left <= 270:
            return depth >= 3 or board.is_finished()
        
        # Between 60 and 20 seconds left, search_depth = 2
        # We need to accelerate the game in order to make sure not to lose because of time
        elif time_left > 20 and time_left <= 60:
            return depth >= 2 or board.is_finished()
        
        # In the last 20 seconds, rapid search (depth = 1)
        # We need to finish fast, no time to search
        else:   
            return depth >= 1 or board.is_finished()
    
    def utility_function(self, state):
        """
        This function gives an evaluation of the board depending on the position of both players.
        """
        board, player = state
        opponent = 1 - player
        
        try:
            steps_player = board.min_steps_before_victory(player)
        except:
            steps_player = None
        try:
            steps_opponent = board.min_steps_before_victory(opponent)
        except:
            steps_opponent = None

        if steps_player == 0:
            return 100000000
        if steps_opponent == 0:
            return -100000000
        
        try:  # The heuristic chosen is simply the difference of the length of the path of both players
            heuristic = board.get_score(player)
        except:
            # if we cannot get the score (noPath exception), we consider the difference between each player's distance to their goals
            heuristic = abs(board.pawns[opponent][0] - board.goals[opponent]) - abs(board.pawns[player][0] - board.goals[player])
        
        return heuristic
        
    
    def max_value(self, state, alpha, beta, depth, game_info, pruning = True):
        """
        MaxValue function for the MiniMax algorithm.
        Alpha/beta pruning to reduce the branching factor.
        """
        if self.isTerminal(state, depth, game_info):
            return self.utility_function(state), None
        value = -math.inf
        action = None
        
        step, time_left = game_info   
        for a, s in self.successors(state, game_info):
            temp_value, _ = self.min_value(s, alpha, beta, depth + 1, game_info)
            if temp_value > value:
                value = temp_value
                action = a
                alpha = max(alpha, value)
            if pruning:
                if value >= beta:
                    return value, action
        return value, action
    
    def min_value(self, state, alpha, beta, depth, game_info, pruning = True):
        """
        MinValue function for the MiniMax algorithm.
        Alpha/beta pruning to reduce the branching factor.
        """
        if self.isTerminal(state, depth, game_info):
            return self.utility_function(state), None
        value = math.inf
        action = None
        
        step, time_left = game_info

        for a, s in self.successors(state, game_info):
            temp_value, _ = self.max_value(s, alpha, beta, depth + 1, game_info)
            if temp_value < value:
                value = temp_value
                action = a
                beta = min(beta, value)
            if pruning:
                if value <= alpha:
                    return value, action
        return value, action
    
    def minimax_search(self, state, game, game_info, pruning = True):
        """
        Main function of the MiniMax algorithm.
        Start with max_value as we want to get the action of maximum value for our player.
        """
        _, action = self.max_value(state, -math.inf, math.inf, 0, game_info)
        return action
    
   
    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in quoridor.py.
        :param player: the player to control in this step (0 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
          eg: ('P', 5, 2) to move your pawn to cell (5,2)
          eg: ('WH', 5, 2) to put a horizontal wall on corridor (5,2)
          for more details, see `Board.get_actions()` in quoridor.py
        """
        #print("percept:", percepts)
        print("player:", player)
        print("step:", step)
        print("time left:", time_left if time_left else '+inf')

        self.player = player
    
        state = (dict_to_board(percepts), player)
        board = dict_to_board(percepts)
        game_info = (step, time_left)
        
        # MiniMax search
        action = self.minimax_search(state, self, game_info)
        flag = board.is_action_valid(action, player)
        print("Action played : ", action)
        
        if flag:
            return action
        else : # In the case where there is an issue in the MiniMax search
            try:
                new_position = board.get_shortest_path(player)[0]
                action = ('P', new_position[0], new_position[1])
            except:
                action = board.get_legal_pawn_moves(player)[0]
            return action
    

if __name__ == "__main__":
    agent_main(MyAgent())
    
    