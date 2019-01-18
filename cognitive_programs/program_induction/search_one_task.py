import cPickle
import time

import numpy as np

from cognitive_programs.program_induction.argument_expansion import \
    CondProgramModel

from cognitive_programs.program_induction.tree_traversal import all_nodes_dfs, \
    first_node_dfs, next_node_dfs


def concept_search_n(concept):

    with open('shared_inputs.pkl', 'rb') as file:
        shared_inputs = cPickle.load(file)

    (
        examples,
        n_examples_to_test,
        prediction_data,
        all_programs,
        model,
        n_progs_to_test,
        it_idx,
        arg_predictor,
        concept_emulator,
        debug,
    ) = shared_inputs

    if debug:
        f = open('{}_{}_notes.txt'.format(it_idx, concept), 'w+')
        f.write('started')
        f.flush()

    arg_log_p, valid_atoms = arg_predictor(concept, prediction_data, all_programs)

    start_time = time.time()

    if debug:
        f.write(str(arg_log_p))
        f.write(str(valid_atoms))
        f.flush()

    cond_model = CondProgramModel(model, arg_log_p, valid_atoms)
    max_desc_length = 12.0
    all_nodes = []
    while len(all_nodes) < n_progs_to_test:
        max_desc_length += 0.5
        all_nodes = zip(*all_nodes_dfs(cond_model.log_T,
                                       cond_model.initial_state, -max_desc_length, cond_model.sub_info))
    nodes_to_consider = sorted(all_nodes)[:n_progs_to_test + 1]
    max_desc_length = nodes_to_consider[-1][0]

    emulator = concept_emulator(concept, examples)
    visited_progs = 0

    current_prog, current_score, search_state = first_node_dfs(
        cond_model.log_T, cond_model.initial_state, -max_desc_length, cond_model.sub_info)
    best_result = (None, np.inf)

    # Initialize last_node_is_ok so that the search can do deep first
    last_node_is_ok = True

    while current_prog != [-1]:
        current_length = -current_score
        visited_progs += 1
        program = cond_model.expand_to_program(current_prog)
        found = emulator.check_program(program, n_examples_to_test)
        if found:
            if current_length < best_result[1]:
                best_result = current_prog, current_length
        else:
            # Program wasn't found, but did it also generate an exception?
            last_node_is_ok = found is not None
        current_prog, current_score, search_state = next_node_dfs(
            search_state, last_node_is_ok)

    current_prog, current_length = best_result

    if debug:
        f.write('complete search')
        f.flush()

    if current_prog is None:
        if debug:
            f.write("nothing found within length={} after searching for{}"
                    "".format(max_desc_length, visited_progs))
            f.flush()
            f.close()
        return None, visited_progs, max_desc_length, 0, None, time.time() - start_time
    else:
        program = cond_model.expand_to_program(current_prog)
        position_in_dl_order = zip(
            *nodes_to_consider)[1].index(current_prog) + 1
        if debug:
            f.write("Found concept={} of length={} after searching{} {} {}"
                    "".format(
                        str(concept), str(current_length), str(visited_progs),
                        str(program),
                        str(position_in_dl_order)))
            f.flush()
            f.close()
        return (program, visited_progs, current_length, position_in_dl_order,
                cond_model.expand_to_program_broken_by_subroutines(
                    current_prog), time.time() - start_time)
