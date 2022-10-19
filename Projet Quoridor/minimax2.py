#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:58:24 2021

@author: augustin
"""

import math

INF = math.inf


def minimax_search(state, game, game_info, pruning = True):
    
    def max_value(state, alpha, beta, depth, game_info):
        if game.isTerminal(state, depth, game_info):
            return game.evaluate(state), None
        value = -INF
        action = None
        for a, s in game.successors(state, game_info):
            temp_value, temp_action = min_value(s, alpha, beta, depth + 1, game_info)
            if temp_value > value:
                value = temp_value
                action = a
                if pruning:
                    if temp_value >= beta:
                        return temp_value, a
                    alpha = max(alpha, temp_value)
        
        return value, action
    
    
    
    def min_value(state, alpha, beta, depth, game_info):
        if game.isTerminal(state, depth, game_info):
            return game.evaluate(state), None
        value = INF
        action = None
        for a, s in game.successors(state, game_info):
            temp_value, temp_action = max_value(s, alpha, beta, depth + 1, game_info)
            if temp_value < value:
                value = temp_value
                action = a
                if pruning:
                    if temp_value <= alpha:
                        return temp_value, temp_action
                    beta = min(beta, temp_value)
        return value, action
    
    value, action = max_value(state, -INF, INF, 0, game_info)
    return action