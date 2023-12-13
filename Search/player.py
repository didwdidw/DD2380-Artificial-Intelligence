#!/usr/bin/env python3
import random
import time
import math

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        start_time = time.time()
        best_move = 0
        depth = 1
        visited_nodes = dict()  # For repeated states checking
        while True:
            try:
                current_best_move = self.deepening_search(
                    initial_tree_node, depth, start_time, visited_nodes)
                best_move = current_best_move
                depth += 1
            except:
                break

        return ACTION_TO_STR[best_move]

    def deepening_search(self, current_node, depth, start_time, visited_nodes):
        children = current_node.compute_and_get_children()
        scores = []
        for child in children:
            scores.append(self.alpha_beta_pruning(
                child, depth, float('-inf'), float('inf'), False, start_time, visited_nodes))

        return children[scores.index(max(scores))].move

    def alpha_beta_pruning(self, current_node, depth, alpha, beta, maximizing_player, start_time, visited_nodes):
        # To break iterative deepening search
        # By the spec, each guess should take less than 75 ms
        if time.time() - start_time > 0.055:
            raise TimeoutError

        current_state = self.state_to_str(current_node.state)
        # Current has been vistied, and it is not possible to dig even deeper
        if current_state in visited_nodes and visited_nodes[current_state][0] >= depth:
            return visited_nodes[current_state][1]

        children = current_node.compute_and_get_children()

        if depth == 0 or len(children) == 0:
            return self.heuristic(current_node)

        if maximizing_player:
            value = float('-inf')
            for child in children:
                value = max(value, self.alpha_beta_pruning(
                    child, depth - 1, alpha, beta, False, start_time, visited_nodes))
                if value >= beta:
                    break  # pruning
                alpha = max(alpha, value)
        else:
            value = float('inf')
            for child in children:
                value = min(value, self.alpha_beta_pruning(
                    child, depth - 1, alpha, beta, True, start_time, visited_nodes))
                if value <= alpha:
                    break  # pruning
                beta = min(beta, value)

        visited_nodes.update({current_state: (depth, value)})
        return value

    def heuristic(self, node):
        score_diff = node.state.player_scores[0] - node.state.player_scores[1]
        best_target = 0
        hookPosition = node.state.hook_positions[0]  # position of my hook
        for fish in node.state.fish_positions:
            # position of each fish
            fishPosition = node.state.fish_positions[fish]
            distance = self.L1Distance(hookPosition, fishPosition)
            if distance == 0:
                return float('inf')
            best_target = max(
                best_target, node.state.fish_scores[fish] * math.exp(-distance))

        return 2*score_diff + best_target

    def L1Distance(self, hookPosition, fishPosition):
        return abs(fishPosition[1] - hookPosition[1]) + min(abs(fishPosition[0] - hookPosition[0]), 20 - abs(fishPosition[0] - hookPosition[0]))

    def state_to_str(self, state):
        return str(state.get_hook_positions()) + str(state.get_fish_positions())
