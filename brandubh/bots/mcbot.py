#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:14:45 2019

@author: john

This file contains classes defining a bot that plays brandubh using a monte
carlo tree search
"""

import math
import random
import copy
from brandubh import Act
from .random_bot import RandomBot
from .greedy_random_bot import GreedyRandomBot


class MCTSBot:
    """
    This bot plays brandubh using a Monte Carlo tree search to select its
    move. An intance of this class is initialised with two parameters.
    1) num_rounds - the number of nodes to be added to the tree
    2) temperature- effects the balance between exploration and exploitation
                    when picking child nodes with uct score
    When an instance is created, and RandomBot object is also created as
    an instance variable. This is used for simulating random games
    
    Monte Carlo tree search:
        This algorithm builds a tree data structure. Each node of the tree
        represents a possible move following the move represented by the
        node's parent node. The root node of the tree is the current board
        position the bot is trying to decide a move for. 
        
        The algorithm starts with a root node and then adds child nodes to
        the root until no more can be added (i.e. all legal moves from the 
        current board position have been added). Once all possible child nodes 
        have been added to a node, the algorithm picks a child node at random 
        (using uct score) to add a child node to.
        
        Each time a child node is added to the tree, a random game is played 
        until the a winner is decided, begining from the board position 
        represented by the new child node. The winner is saved in the node
        and passed to all parent nodes. So that each node in the tree has a
        record of how many random games, beginning from the corresponding
        board position, the black or white player won. The ratio of black wins 
        vs white wins gives a way of ranking how good a board position is for 
        a particular player.
        
        This continues until a given number of nodes ('num_rounds') are
        added to the tree.
        
        The child node of the root with the best ranking is selected as the
        next move.
    """
    is_trainable = False
    
    def __init__(self, num_rounds=1400, temp=1.4, use_greedy_rand=False):
        self.num_rounds = num_rounds
        self.temperature = temp
        self.bot = GreedyRandomBot() if use_greedy_rand else RandomBot()
        self.root = None
    
    def select_move(self, game_state, return_root=False):
        """
        This method uses the Monte Carlo tree search to select what move
        to make next given the board position in game_state.
        """
        root = MCTSNode(game_state.copy())
        
        # add num_rounds nodes to the tree.
        for i in range(self.num_rounds):
            # To add a child node, begin at the root of the tree.
            node = root
            
            # While child nodes can't be added to the current node and
            # the current node doesn't represent a game state where the
            # game is over, select a child as the current node using uct
            while (not node.can_add_child()) and (not node.is_terminal()):
                if node.children == []:
                    break
                node = self.select_child(node)
                
            # Add a random child node if possible
            if node.can_add_child():
                node = node.add_random_child()
                
            # Simulate a random game from the current board position, record
            # the winner and pass it back to all parent nodes
            winner = self.simulate_random_game(node.game_state)
            while node is not None:
                node.record_win(winner)
                node = node.parent
         
        # Once 'num_rounds' nodes have been added to the tree, select the 
        # child node of the root with the best win rate as the next move
        best_move = None
        best_frac = -1
        for child in root.children:
            child_frac = child.winning_frac(game_state.player)
            if child_frac > best_frac:
                best_frac = child_frac
                best_move = child.move
    
        output = Act.pass_turn() if best_move is None else Act.play(best_move)
        if return_root:
            output = (output, root)
        return output
    
    def select_child(self, node):
        """
        This method selects a child with the best uct score
        """
        # total_rollouts = sum(child.num_rollouts for child in node.children)
        total_rollouts = node.num_rollouts
        best_score = -1
        best_child = None
        
        for child in node.children:
            score = uct_score(total_rollouts, child.num_rollouts,
                              child.winning_frac(node.game_state.player),
                              self.temperature)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def simulate_random_game(self, game_state):
        """
        This method plays a game of brandubh begining from the board position
        in game_state and plays until a winner is decided. At each turn,
        moves are selected at random. The method returns the winner of the
        game when it is over.
        """
        game = copy.deepcopy(game_state)
        
        while game.is_not_over():
            random_move = self.bot.select_move(game)
            game.take_turn(random_move)
            
        return game.winner
    
    
class MCTSNode:
    """
    This class is a node of a tree used in Monte Carlo tree search. Instance
    variables include:
        * game_state containing the board position the node represents
        * parent - a link to the parent node
        * move - the move that created the board position from the parent
        * win_counts - a dictionary containg number of wins for blacl/white
        * num_rollouts - number of random games that stemmed from this node
        * children - a list of child nodes
        * unvisited_moves - a list of possible child nodes to add
    """
    
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {-1: 0,
                            1: 0}
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.legal_moves()    
    
    def add_random_child(self):
        """
        This method adds a random child to self and returns the added child
        """
        index = random.randint(0, len(self.unvisited_moves)-1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.copy()
        new_game_state.take_turn(Act.play(new_move))
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node
        
    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1
    
    def can_add_child(self):
        return len(self.unvisited_moves) > 0
    
    def is_terminal(self):
        return not self.game_state.is_not_over()
    
    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)
    
    def corresponds_to(self, history_link):
        if history_link:
            if self.game_state.player == history_link.player:
                if self.game_state.game_set.board == history_link.board:
                    return True
        return False
    


def uct_score(parent_rollouts, child_rollouts, win_frac, temp):
    """
    The uct score (Upper Confidence bounds applied to Trees) is given by
    the formula:
        s = w/n + c*sqrt( ln(N)/n ),
    where 
    w = number of wins for the current player in games
        stemming from the current node, 
    n = number of games stemming from the current node
    N = number of games stemming from the parent node
    c = temperature
    
    The first term is called exploitation (large for child nodes where the
    player wins alot)
    The second term is called exploration (large for child nodes with few
    random games stemming from them)
    """
    exploration = math.sqrt( math.log(parent_rollouts) / child_rollouts )
    return win_frac + temp*exploration