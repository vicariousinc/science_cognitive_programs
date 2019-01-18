import numba as nb
import numpy as np

from cognitive_programs.program_induction.program_compression import \
    expand_to_atoms


@nb.njit
def syntax_check(node_compressed, sub_info, partial=False):
    """Partial allows to check if what we have is, potentially, the prefix of a valid program"""
    # default argument for sub_info: empty_sub_info = (np.array([], dtype=int), np.array([], dtype=int), 1000000000000000000)
    # Expand subroutines
    node = expand_to_atoms(node_compressed, sub_info)
    in_loop = False
    current_loop_has_1 = False
    current_loop_has_14 = False
    for inst in node:
        if inst == 25:
            if not in_loop:
                in_loop = True
            else:
                return False
        elif inst == 26:
            if in_loop and current_loop_has_1 and current_loop_has_14:
                in_loop = current_loop_has_1 = current_loop_has_14 = False
            else:
                return False
        elif inst == 1 and in_loop:
            current_loop_has_1 = True
        elif inst == 14 and in_loop:
            current_loop_has_14 = True
        elif inst == 16 and in_loop:
            return False
    return not in_loop or partial


@nb.njit
def first_node_dfs(log_T, initial_state, min_score, sub_info, max_depth=1000000000000000000, maxtraversals=1000000000000000000):
    min_score = float(min_score)  # make sure numba knows this is a float (otherwise, sometimes, it doesn't (bug in numba))
    order = np.zeros(log_T.shape, np.int64)
    for i in xrange(order.shape[1]):
        order[i] = (-log_T[i]).argsort()
    node = [order[initial_state, 0]]  # most likely first node
    node_idx = [0]
    it = 0

    # score and return current node if adequate
    score = log_T[initial_state, node[0]]
    for p in xrange(1, len(node)):
        score += log_T[node[p - 1], node[p]]
    if min_score <= score:
        search_state = log_T, initial_state, min_score, max_depth, maxtraversals, list(node), list(node_idx), it, order, score, sub_info
        return list(node), score, search_state  # the invocation to list here is to make a copy, don't remove!


@nb.njit
def next_node_dfs(search_state, last_node_is_ok):
    """ Assumes that the sequence starts with initial_state and continues from there,
        returns the sequences minus that initial state.
    """
    log_T, initial_state, min_score, max_depth, maxtraversals, node, node_idx, it, order, score, sub_info = search_state
    min_score = float(min_score)  # make sure numba knows this is a float (otherwise, sometimes, it doesn't (bug in numba))
    n_states = log_T.shape[0]
    if it == maxtraversals:
        assert False, "Number of traversals exceeded"
    while True:
        # next node ##
        # try adding a value at the end
        for next_idx, next_state in enumerate(order[node[-1]]):
            if last_node_is_ok and min_score <= score + log_T[node[-1], next_state] and len(node) < max_depth \
                    and syntax_check(np.array(node + [next_state]), sub_info, partial=True):
                node.append(next_state)
                node_idx.append(next_idx)
                break
        # adding a value at the end failed, so we are a leave
        else:
            for p in xrange(len(node) - 1, -1, -1):
                if node_idx[p] != n_states - 1:  # find where within the node to increase (and discard all others after)
                    old_idx = node_idx[p]
                    del node_idx[p:]
                    del node[p:]
                    node_idx.append(old_idx + 1)
                    prev_state = node[p - 1] if p > 0 else initial_state
                    node.append(order[prev_state, node_idx[p]])
                    break
            else:
                search_state = log_T, initial_state, min_score, max_depth, maxtraversals, list(node), list(node_idx), it, order, score, sub_info
                return [-1], score, search_state   # end of the generator, can't increase even the root
        last_node_is_ok = True  # We can now make progress again, regardless of whether we could at the beginning
        it += 1
        # score and return current node if adequate
        score = log_T[initial_state, node[0]]
        for p in xrange(1, len(node)):
            score += log_T[node[p - 1], node[p]]
        if min_score <= score and syntax_check(np.array(node), sub_info, partial=False):
            search_state = log_T, initial_state, min_score, max_depth, maxtraversals, list(node), list(node_idx), it, order, score, sub_info
            return list(node), score, search_state  # the invocation to list here is to make a copy, don't remove!


@nb.njit
def all_nodes_dfs(log_T, initial_state, min_score, sub_info, max_depth=1000000000000000000, maxtraversals=1000000000000000000):
    """ Assumes that the sequence starts with initial_state and continues from there,
        returns the sequences minus that initial state.
    """
    # default argument for sub_info: empty_sub_info = (np.array([], dtype=int), np.array([], dtype=int), 1000000000000000000)
    min_score = float(min_score)  # make sure numba knows this is a float (otherwise, sometimes, it doesn't (bug in numba))
    order = np.zeros(log_T.shape, np.int64)
    for i in xrange(order.shape[1]):
        order[i] = (-log_T[i]).argsort()
    n_states = log_T.shape[0]
    node = [order[initial_state, 0]]  # most likely first node
    node_idx = [0]
    lengths_dfs = [-1.0]
    nodes_dfs = [[-1, ]]
    for it in xrange(maxtraversals):
        # score and return current node if adequate
        score = log_T[initial_state, node[0]]
        for p in xrange(1, len(node)):
            score += log_T[node[p - 1], node[p]]
        if min_score <= score and syntax_check(np.array(node), sub_info, partial=False):
            lengths_dfs.append(-score)
            nodes_dfs.append(list(node))
        # next node ##
        # try adding a value at the end
        for next_idx, next_state in enumerate(order[node[-1]]):
            if min_score <= score + log_T[node[-1], next_state] and len(node) < max_depth \
                    and syntax_check(np.array(node + [next_state]), sub_info, partial=True):
                node.append(next_state)
                node_idx.append(next_idx)
                break
        # adding a value at the end failed, so we are a leave
        else:
            for p in xrange(len(node) - 1, -1, -1):
                if node_idx[p] != n_states - 1:  # find where within the node to increase (and discard all others after)
                    old_idx = node_idx[p]
                    del node_idx[p:]
                    del node[p:]
                    node_idx.append(old_idx + 1)
                    prev_state = node[p - 1] if p > 0 else initial_state
                    node.append(order[prev_state, node_idx[p]])
                    break
            else:
                break  # end of the generator, can't increase even the root
    else:
        assert False, "Number of traversals exceeded"

    return lengths_dfs[1:], nodes_dfs[1:]
