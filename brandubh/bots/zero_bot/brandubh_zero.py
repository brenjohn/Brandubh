#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:49:08 2020

@author: john

This file contains classes used to create a bot that can play brandubh using
the AlphaGo-Zero approach.
"""
import numpy as np
import os

from ...game import Act
from .zero_network import ZeroNet


class ZeroBot:
    """
    The ZeroBot uses the policy used by AlphaGoZero to select moves. Namely,
    it uses a type of Monte Carlo tree search which has been integrated with
    a neural network that evaulates board positions and predicts which
    branches of the search tree will be visted most often. The move it selects
    is the move corresponding to the most visited branch during this tree 
    search.
    
    Arguments:
        evals_per_turn : 
            The number of nodes that will be added to the tree structure when 
            selecting a move. For larger num_rounds, the bot will take longer 
            to choose a move but it should also pick stringer moves.
            
        batch_size :
            The number of tree nodes to evaluate at any one time.
        
        c :
            A parameter to balance exploration and exploitation. The bot will 
            explore more for larger c.
                     
        alpha :
            The dirichlet noise parameter. The noise is used to increase the 
            probabilty random moves get explored during move selection.
                     
        sampling_turns :
            The number of turns at the start of a game where moves will be
            sampled from a probability distribution defined by the move visit
            count statitics in the tree. After this number of turns, the move
            with the largest visit count is selected.
            
        network :
            The neural network to use.
    """
    
    def __init__(
            self, 
            evals_per_turn = 7000, 
            batch_size = 70,
            c = 1.4,
            alpha = 0.15,
            sampling_turns = 6,
            network = None
        ):
        self.evals_per_turn = evals_per_turn
        self.batch_size = batch_size
        self.c = c
        self.alpha = alpha
        self.sampling_turns = sampling_turns
        
        if network:
            self.network = network
        else:
            self.network = ZeroNet()
                
        self.compile_network((1.0, 0.1))
        
        self.climbers = [TreeClimber(c) for i in range(batch_size)]
        self.root = None
    
    
    def select_move(
            self, 
            game_state,
            reuse_search_tree=True
        ):
        """
        Select a move to make from the given board position (game_state).
        
        The algorithm uses a combination of a neural network and a Monte Carlo
        tree search to search the decision tree stemming from the given board
        position. It returns the move associated with the most visited branch 
        stemming from the root.
        
        This method creates a tree structure representing the search history
        of the algorithm and is used to save evaluations of board positions
        and statistics regarding nodes visited.
        
        If return_visit_counts is true, the distribution of visits over the 
        branches of the root in the search tree will be returned along with
        the selected move. This distribution can be used to train the neural
        network.
        """
        
        # If a search tree is already saved, reuse the subtree relevant to
        # the given game state. Otherwise, start with a tree consisting of a 
        # root node only. The root node is associated with the given board 
        # position.
        if reuse_search_tree:
            self.update_root_to_current_game_state(game_state)
            
        if self.root is None:
            self.root = self.create_root_node(game_state.copy())
        self.root.add_noise(self.alpha)
        
        
        # If no legal moves can be made from the given board position, pass 
        # the turn. This happens when all of the players pieces are surrounded,
        # if the player has no pieces left or if the game is over. 
        if not self.root.branches:
            return Act.pass_turn()
        
        # Run the hybrid neural network - Monte Carlo tree search algorithm to
        # update the current search tree with new board evaluations.
        self.update_tree()
        
        # Select one of the the possible moves using visit count statistics
        # from the tree.
        act = self._select_move(num_turns = game_state.num_moves)
        return act
    
    
    def _select_move(self, num_turns):
        """
        Creates a list of possible moves and selects a one with one of the 
        following methods:
        
        1) If the number of turns in the game is less than a certain 
        threshold, the move is randomly sampled from the prob dist defined by 
        the visit counts.
        
        2) Otherwise, the move with the highest visit count is selected.
        """
        moves = [move for move in self.root.moves()]
        if moves:
            if num_turns < self.sampling_turns:
                move_distribution = np.asarray([
                    self.root.branches[move].visit_count for move in moves
                ])
                move_distribution = move_distribution / sum(move_distribution)
                move_ind = np.random.choice(len(moves), p=move_distribution)
                move = moves[move_ind]
            else:
                move = max(moves, key=self.root.visit_count)
            
            # Return the move as an Act.
            return Act.play(move)
        
        # If no legal move is found then pass the turn.
        return Act.pass_turn()
        
    
    def update_root_to_current_game_state(self, game_state):
        """
        Attempts to reuse the existing search tree by finding the current 
        game_state within the tree's descendants.
        """
        # If a search tree is saved.
        if self.root is None:
            return
        
        # Collect the moves made since the last turn taken.
        moves_since_last_turn = []
        historic_state = game_state.history
        root_found = False
        
        for _ in range(3):
            if self.root.corresponds_to(historic_state):
                root_found = True
                break
            if historic_state.previous_state is not None:
                moves_since_last_turn.insert(0, historic_state.last_move)
                historic_state = historic_state.previous_state
            else:
                break
        
        # Set the root to None if it isn't in the local history.
        if not root_found:
            self.root = None
            return
        
        # Update the root with the moves made since the last turn.
        for move in moves_since_last_turn:
            if self.root.has_child(move):
                self.root = self.root.get_child(move)
            else:
                # Set root to none if a relevant subtree doesn't exist.
                self.root = None
                return
            
        # Disconnect the root from any parent it might have.
        if self.root is not None:
            self.root.parent = None
    
    
    def update_tree(self):
        """
        Runs the hybrid Monte Carlo - neural network tree search to populate
        the search tree with board evaluations.
        """
        next_states = [None] * len(self.climbers)
        for climber in self.climbers:
            climber.set_node(self.root)
        
        evals_made = 0
        while evals_made != self.evals_per_turn:
            
            climbers_ready = 0
            while (climbers_ready < self.batch_size and 
                   evals_made < self.evals_per_turn):
                
                climber = self.climbers[climbers_ready]
                climber.climb_up()

                # TODO: Can possibly avoid extra copying of terminal states by
                # storing states inside branches.
                next_state = climber.get_next_state()
                if next_state.is_not_over():
                    next_states[climbers_ready] = next_state
                    climbers_ready += 1
                else:
                    climber.evaluate_terminal_leaf()
                    climber.climb_down()
                    
                evals_made += 1
                    
            climbers = self.climbers[0:climbers_ready]
            states = next_states[0:climbers_ready]
            if climbers_ready > 0:
                predictions = self.network.predict(states)
            
                for state, pred, climber in zip(states, predictions, climbers):
                    climber.expand_branch(state, *pred)
                    climber.climb_down()

                
    def create_root_node(self, game_state, move=None, parent=None):
        """
        This method creates a tree node for the given board position and adds
        it to the tree structure. It will be linked to the given parent node
        and the given move is stored as the last move taken to produce the
        given game state. This is useful for trversing and updating the tree 
        structure when other nodes are added to it.
        """
        # Pass the game state to the neural network to both evaluate the 
        # how good the board position is and get the prior probability 
        # distribution over possible next moves (ie the predicted distribution 
        # of visit counts).
        prediction = self.network.predict([game_state])
        move_priors, value = prediction[0]
        
        # Create the node for the given game state, with the predicted value
        # and priors, and attach it to the tree.
        new_node = TreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
        
        
    def turn_on_eval_mode(self, look_ahead = None, **kwargs):
        """
        """
        self.old_alpha = self.alpha
        self.old_evals_per_turn = self.evals_per_turn
        
        self.alpha = 0.0
        if look_ahead is not None:
            self.evals_per_turn = look_ahead
    
    
    def turn_off_eval_mode(self):
        """
        """
        self.alpha = self.old_alpha
        self.evals_per_turn = self.old_evals_per_turn
    
    
    def save_bot(self, prefix="model_data/"):
        """
        Method to save the attributes of the current bot and the weights of
        its neural network under the directory given by the parameter 
        'prefix'
        """
        if not os.path.exists(prefix):
            os.makedirs(prefix)
            
        network_load_command = self.network.save_network(prefix)
        attributes = {
            "evals_per_turn" : self.evals_per_turn,
            "c"              : self.c,
            "batch_size"     : self.batch_size,
            "alpha"          : self.alpha,
            "sampling_turns" : self.sampling_turns,
            "network_load_command": network_load_command
        }
        
        np.save(prefix + "model_attributes.npy", attributes)
        
    
    def load_bot(self, prefix="model_data/"):
        """
        Method to load the attributes and neural network saved under the given
        directory.
        """
        attributes = np.load(prefix + "model_attributes.npy",
                             allow_pickle='TRUE').item()
        
        self.evals_per_turn = attributes["evals_per_turn"]
        self.c = attributes["c"]
        self.batch_size = attributes["batch_size"]
        self.climbers = [TreeClimber(self.c) for i in range(self.batch_size)]
        self.alpha = attributes["alpha"]
        
        network_load_command = attributes["network_load_command"]
        exec(network_load_command)
        self.network.load_network(prefix)
    
        
    def compile_network(self, loss_weights):
        self.network.compile_network(*loss_weights)
    
    
    def get_DataManager(self, max_bank_size = 0):
        """
        Return a manager object to manage generated training data for the
        bot's neural network.
        """
        if max_bank_size:
            return self.network.get_DataManager(max_bank_size)
        return self.network.get_DataManager()



class Branch:
    """
    Instances of this class are used to store statistics gathered, by the
    ZeroBot select_move algorithm, on how often a branch of the decision tree
    stemming from a particular gamestae was visited and the estimated value
    of the resulting board position. It also saves the prior probability of
    the move associated with an instance of Branch as predicted by the neural
    network.
    """
    def __init__(self, prior):
        self.virtual_loss = 0
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0 # when divided by visit count should give the
                             # average value of the board corresponding to this
                             # branch

        
        
class TreeNode:
    """
    This class can represent a node (corresponding to a game state) in the 
    decision tree stemming from a game state of brandubh.
    
    Instances of this class are used to build a tree structure to record the
    search history of the ZeroBot select_move algorithm. It saves an instance
    of the game state it represents, the expected value of that game state as
    predicted by the neural network, a reference to its parent node if it has
    one and a tuple representing the previous move of the game corresponding 
    game state which created it.
    
    It also contains two dictionaries, indexed by game moves, which hold 
    references to any child nodes attached to the current instance in the tree
    structure and branch objects containing statistics regarding the search 
    history of the select_move method.
    """
    def __init__(self, game_state, value, priors, parent, last_move):
        self.state = game_state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        
        # Used when selecting a branch stemming from this node. The creation 
        # of the node counts as the first visit so is initialised to 1.
        self.total_visit_count = 1
        
        self.branches = {
            move : Branch(prior) for move, prior in priors.items()
        }
                
        self.children = {}
         
    def moves(self):
        return self.branches.keys()
    
    def add_child(self, move, child_node):
        self.children[move] = child_node
        
    def has_child(self, move):
        return move in self.children
    
    def get_child(self, move):
        return self.children[move]
    
    def add_noise(self, alpha):
        if alpha > 0:
            num_branches = len(self.branches)
            noise = np.random.gamma(alpha, 1, num_branches)
            N = sum(noise)
            for i, branch in enumerate(self.branches.values()):
                branch.prior = (0.75 * branch.prior + 0.25 * noise[i] / N)
    
    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0
        return (branch.total_value + branch.virtual_loss) / branch.visit_count
    
    def branch_score_stats(self, move):
        branch = self.branches[move]
        visit_count = branch.visit_count
        prior = branch.prior
        if visit_count:
            expected_value = (branch.total_value + branch.virtual_loss) / branch.visit_count
        else:
            expected_value = 0
        return expected_value, prior, visit_count
    
    def increment_virtual_loss(self, move):
        if move:
            branch = self.branches[move]
            branch.virtual_loss -= 1
            branch.visit_count += 1
        
    def decrement_virtual_loss(self, move):
        if move:
            branch = self.branches[move]
            branch.virtual_loss += 1
            branch.visit_count -= 1
    
    def lock_branch(self, move):
        # TODO: not locking a None move means there's a chance the 
        # corresponding node will be evaluated more than once, possibly 
        # skewing the tree statistics slightly.
        if move:
            # Other PUCT scores will never be this negative
            self.branches[move].virtual_loss = -700
            self.branches[move].visit_count += 1
        
    def unlock_branch(self, move):
        if move:
            self.branches[move].virtual_loss = 0
            self.branches[move].visit_count -= 1
    
    def prior(self, move):
        return self.branches[move].prior
    
    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
    
    def record_visit(self, move, value):
        self.total_visit_count += 1
        # If the move isn't a pass
        if move:
            self.branches[move].visit_count += 1
            self.branches[move].total_value += value
            
    def is_not_terminal_leaf(self):
        return self.state.is_not_over()
            
    def corresponds_to(self, history_link):
        if history_link:
            player, game_set = self.state.player, self.state.game_set
            return history_link.corresponds_to(player, game_set)
        return False
    
    def check_legality(self):
        local_checks = [self.state.is_move_illegal(m) 
                        for m in self.branches.keys()]
        
        descendant_checks = []
        if self.state.is_not_over():
            for child in self.children.values():
                descendant_checks += child.check_legality()
        
        return descendant_checks + local_checks
            
            
            
class TreeClimber:
    """
    An instance TreeClimber is responsible for traversing a search tree
    according to the PUCT (polynomial upper confidence tree) rule.
    
    A virtual loss, stored in the branch objects, is used to modify the PUCT
    score of branches traversed by a treeclimber to discourage different 
    instances of TreeClimber from exploring the same branches. This 
    facilitates batching of network predictions.
    
    Expanding a leaf node of a search tree happens by the following steps:
        
        1 - A treeclimber initialised with the root node uses the climb_up
        method to traverse up the tree to a leaf node. The virtual loss
        of traversed branches is increased during this process.
        
        2 - The leaf can be expanded with the expand_branch method
        
        3 - The treeclimber then traverses back to the root node using the
        climb_down method which also updates the branch fields along the way
        with the appropriate values. Changes to the virtual loss are undone
        here.
    """
    def __init__(self, c):
        self.c = c
        self.node = None
        self.next_move = None
        self.value = None
        
    def set_node(self, node):
        self.node = node
        
    def get_next_state(self):
        if self.next_move:
            action = Act.play(self.next_move)
        else:
            # If the current player can't make any moves from the
            # selected game state then next move will be 'None' meaning
            # the player passes the turn.
            action = Act.pass_turn()
        next_state = self.node.state.copy()
        next_state.take_turn_with_no_checks(action)
        return next_state
        
    def branch_can_be_expanded(self):
        return self.node.is_not_terminal_leaf()
        
    def climb_up(self):
        """
        climb up the tree to a leaf node and select a move to make from the 
        corresponding leaf game state.
        """
        node = self.node
        next_move = self.select_branch()
        
        while node.has_child(next_move):
            node.increment_virtual_loss(next_move)
            node = node.get_child(next_move)
            self.node = node
            next_move = self.select_branch()
            
        node.lock_branch(next_move)
        self.next_move = next_move
    
    def climb_down(self):
        """
        Climb down the tree and update the nodes traversed to get to the leaf 
        node with the new value for the new move.
        """
        node = self.node
        move = self.next_move
        value = self.value
        
        node.unlock_branch(move)
        while node.parent is not None:
            node.record_visit(move, value)
            move = node.last_move
            node = node.parent
            node.decrement_virtual_loss(move)
            # TODO: explore using -0.9 here instead.
            value *= -1
            
        node.record_visit(move, value)
        self.node = node
        
    def evaluate_terminal_leaf(self):
        # If the game in the current state is over, then the last
        # player must have won the game. Thus the value/reward for the
        # other player is 1. The current node is not updated with
        # the new reward as no branches can stem from a finished game
        # state.
        
        # self.next_move = self.node.last_move
        # self.node = self.node.parent
        self.value = 1
        
    def expand_branch(self, state, priors, value):
        # Create the node for the given game state, with the predicted value
        # and priors, and attach it to the tree.
        new_node = TreeNode(state, value, priors, self.node, self.next_move)
        self.node.add_child(self.next_move, new_node)
        self.value = -1 * value
    
    def select_branch(self):
        """
        This method selects a move/branch stemming from the given node by 
        picking the move that maximises the following PUCT score:
            
            Q + c*p*sqrt(N)/(1+n),
            
        where Q = the estimated expected reward for the move,
              c = a constant balancing exploration-exploitation,
              p = prior probability for the move,
              N = The total number of visits to the given node
              n = the number of those visits that went to the branch 
                  associated with the move
                  
        Christopher D. Rosin - Multi-armed Bandits with Episode Context
        """
        self.c_sqrt_total_n = np.sqrt(self.node.total_visit_count) * self.c
        
        moves = self.node.moves()
        if moves:
            return max(moves, key=self.branch_score)
        else:
            # If moves is empty then no legal moves can be made from the game
            # state corresponding to the given node.
            return None
        
    def branch_score(self, move):
        q, p, n = self.node.branch_score_stats(move)
        return q + p * self.c_sqrt_total_n/(1+n)
        