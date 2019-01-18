import argparse
import copy
import cPickle
import gzip
import time

import numpy as np
import parmap

from cognitive_programs.arg_prediction.eval_classifier_cnn import predict_arg_order_all_tasks_with_prob
from cognitive_programs.program_induction import arg_predictors
from cognitive_programs.program_induction.oracle import Oracle
from cognitive_programs.program_induction.program_compression import ProgramModel
from cognitive_programs.program_induction.search_one_task import concept_search_n
from cognitive_programs.program_induction.utils import create_commands_dict
from cognitive_programs.tools.system import SYMBOLIC_DATA_PATH, TARGET_PROGRAMS_PATH


global_EC_round = 0
global_results = {}
global_subroutines_enabled = False


def save_results(concept_name, prog, desc_len, visited_nodes,
                 position_in_dl_order, time_taken, concept_emulator,
                 arg_predictor, num_exs, n_programs_to_search, order):
    emul = "real" if concept_emulator != Oracle else "oracle"

    if arg_predictor == arg_predictors.ground_truth_arg_predictor:
        arg = "_ground_truth"
    elif arg_predictor == arg_predictors.cnn_arg_predictor:
        arg = "_with_arg_pred"
    elif arg_predictor == arg_predictors.uniform_arg_predictor:
        arg = "_with_uniform_pred"
    elif arg_predictor == arg_predictors.mix_fixation_provided_and_cnn_arg_predictor:
        arg = "_mix_arg_pred_and_ground_truth_fixation"
    else:
        assert False, "Unkown arg predictor"
    filename = "exp_" + emul + arg + "_order_" + \
        str(order) + "_" + str(n_programs_to_search) + "_numExs=" + \
        str(num_exs) + ".npz"
    global_results[concept_name] = (
        prog, desc_len, visited_nodes, position_in_dl_order, global_EC_round,
        global_subroutines_enabled, time_taken)
    np.savez_compressed(filename, global_results)


def search_all_concepts(model,
                        examples,
                        prediction_data,
                        all_programs, found_programs,
                        n_programs_to_search,
                        it,
                        arg_predictor,
                        concept_emulator,
                        order=1,
                        n_examples_to_test=10,
                        debug=False):
    """ Iterative deepening"""
    found_programs = copy.deepcopy(found_programs)

    concepts_to_search = [
        c for c in all_programs.keys() if c not in found_programs]

    shared_inputs = (
        examples,
        n_examples_to_test,
        prediction_data,
        all_programs,
        model,
        n_programs_to_search,
        it,
        arg_predictor,
        concept_emulator,
        debug,
    )

    print "Saving shared inputs"
    with open('shared_inputs.pkl', 'wb') as file:
        cPickle.dump(shared_inputs, file)

    results = parmap.map(concept_search_n, concepts_to_search)

    min_searched_progs = 1000000000
    max_searched_progs = 0
    for concept_name, (prog, visited, desc_len, position_in_dl_order,
                       prog_by_sub, time_taken) in zip(concepts_to_search, results):
        if prog is None:
            min_searched_progs = min(min_searched_progs, visited)
            max_searched_progs = max(max_searched_progs, visited)
        else:
            save_results(concept_name, prog_by_sub, desc_len, visited,
                         position_in_dl_order, time_taken, concept_emulator,
                         arg_predictor,
                         n_examples_to_test, n_programs_to_search, order)
            found_programs[concept_name] = prog
    if debug:
        print "Planned to search %d programs, for not found concepts this " \
              "resulted in between %d and %d emulator executions (due to " \
              "pruning)" % (n_programs_to_search, min_searched_progs, max_searched_progs)
        print "Found %d out of %d programs" % (len(found_programs), len(all_programs))
    return found_programs


def program_search_example(n_programs_to_search,
                           all_programs,
                           found_programs,
                           examples,
                           prediction_data,
                           order,
                           arg_predictor,
                           concept_emulator,
                           n_examples_to_test=10,
                           explore_mode='subroutine_disable_then_enable',
                           max_iters=100,
                           debug=False):
    global global_EC_round
    global global_subroutines_enabled

    command_set = set(create_commands_dict().keys())
    n_atoms = max(command_set) + 2
    # Avoid exploring nonexistent commands
    no_pseudo_count_to = list(set(range(max(command_set) + 1)) - command_set)

    open_instr, close_instr = [25], [26]
    if order == 0:
        pseudo_counts = [15, 15]
    elif order == 1:
        pseudo_counts = [1e-1, 1e-1]

    for concept_name, prog in found_programs.iteritems():
        save_results(concept_name, prog, 0.0, 0, 0, 0,
                     concept_emulator, arg_predictor,
                     n_examples_to_test, n_programs_to_search, order)

    if debug:
        print found_programs, len(found_programs)
    # Explore - Compress steps
    start_time = time.time()

    if explore_mode == 'subroutine_enable_then_disable':
        subroutine_enable_modes = [True, False]
        pseudo_counts = [1e-1, 1e-1]
    elif explore_mode == 'subroutine_disable_then_enable':
        subroutine_enable_modes = [False, True]
        pseudo_counts = [1e-1, 1e-1]
    elif explore_mode == 'only_subroutine_disable':
        subroutine_enable_modes = [False]
        pseudo_counts = [1e-1]
    elif explore_mode == 'only_subroutine_enable':
        subroutine_enable_modes = [True]
        pseudo_counts = [1e-1]
    else:
        raise ValueError("Invalid input of exploration_mode")

    for it_outer in xrange(1):
        for subroutines_enabled_mode, pseudo_count in zip(subroutine_enable_modes, pseudo_counts):
            num_tasks_solved_so_far = len(found_programs)
            for it_inner in xrange(max_iters):
                global_EC_round = it_inner
                global_subroutines_enabled = subroutines_enabled_mode
                if debug:
                    print "#################### ITER ", it_inner
                # Compress
                found_progr_instr = list(
                    set([zip(*p)[0] for p in found_programs.values()]))
                model = ProgramModel(found_progr_instr, n_atoms, open_instr, close_instr, pseudo_count,
                                     no_pseudo_count_to=no_pseudo_count_to)
                max_subroutines = 100 if subroutines_enabled_mode else 0
                model.compress(
                    order=order, max_subroutines=max_subroutines, debug=debug)

                np.savez_compressed(
                    'model_log_T_ordre={}_argPred={'
                    '}_maxVisitedNodes={}_numExs={'
                    '}_subroutineEnable={}_innerIt={'
                    '}.npz'.format(
                        order, str(arg_predictor), n_programs_to_search,
                        n_examples_to_test, subroutines_enabled_mode, it_inner),
                    model.log_T)

                longest_program = max(model.program_lengths())
                if debug:
                    print "Largest cost of programs found so far:", longest_program
                    print "Time so far", (time.time() - start_time) / 60.0, "minutes"
                # Explore
                new_programs = search_all_concepts(model,
                                                   examples,
                                                   prediction_data,
                                                   all_programs,
                                                   found_programs,
                                                   n_programs_to_search,
                                                   it_inner,
                                                   arg_predictor,
                                                   concept_emulator,
                                                   order,
                                                   n_examples_to_test,
                                                   debug)
                if new_programs == found_programs:
                    if debug:
                        print "EC cycle saturated on iteration", it_inner, "with subroutines", "enabled" if subroutines_enabled_mode else "disabled"
                    break
                found_programs = new_programs
                if debug:
                    print "len new programs", len(new_programs)
                    print "len all programs", len(all_programs)
                    print "Time so far", (time.time() - start_time) / 60.0, "minutes"

            if len(new_programs) == num_tasks_solved_so_far:
                if debug:
                    print "No more tasks found in the mode SubroutineEnable={" \
                          "}_outerIt={}_interIt={}".format(
                              subroutines_enabled_mode, it_outer, it_inner)
                return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Search for programs consistent with example input/output pairs")
    parser.add_argument('--arg_prediction_type', dest='arg_prediction_type', type=str, default="ground_truth_arg")
    parser.add_argument('--target_concept', dest='target_concept', type=str, default=None, help="Concept to be tested")
    parser.add_argument('--learnt_progs_data_path', dest='learnt_progs_data_path', type=str, default=None)
    parser.add_argument('--n_progs', dest='n_programs_to_search', type=int, default=1000000)
    parser.add_argument('--explore_mode', dest='explore_mode', type=str, default='subroutine_disable_then_enable', help="Whether to enable subroutine first or later, or only one mode")
    parser.add_argument('--order', dest='order', type=int, default=1)
    parser.add_argument('--use_real_emulator', dest='use_real_emulator', default=False, action='store_true', help='Use the real emulator, instead of the default Oracle')
    parser.add_argument('--num_iterations', dest='num_iterations', type=int, default=100)

    options = parser.parse_args()

    with gzip.GzipFile(SYMBOLIC_DATA_PATH, 'rb') as file:
        examples = cPickle.load(file)
    with gzip.GzipFile(TARGET_PROGRAMS_PATH, 'rb') as file:
        all_programs = cPickle.load(file)

    if options.learnt_progs_data_path is not None:
        learnt_progs_data = np.load(options.learnt_progs_data_path)['arr_0'].tolist()
        learnt_progs = {}
        subroutine_enable = False

        for task_name, data in learnt_progs_data.iteritems():
            if len(data[0][0]) == 1:
                prog = list(sum(data[0], ()))
            else:
                prog = data[0]

            learnt_progs[task_name] = prog
    else:
        learnt_progs = {k: v for k, v in all_programs.iteritems()
                        if len(v) <= 6}

    if options.target_concept is not None:
        all_programs = {options.target_concept: all_programs[options.target_concept]}
        examples = {options.target_concept: examples[options.target_concept]}

    if options.use_real_emulator:
        try:
            from cognitive_programs.program_induction.real_emulator import RealEmulator
        except ImportError:
            print "Unable to import RealEmulator, try running without --use_real_emulator to use the Oracle"
            raise
        concept_emulator = RealEmulator
    else:
        concept_emulator = Oracle

    if options.arg_prediction_type == 'ground_truth_arg':
        arg_predictor = arg_predictors.ground_truth_arg_predictor
        prediction_data = None
    elif options.arg_prediction_type == 'uniform_arg':
        arg_predictor = arg_predictors.uniform_arg_predictor
        prediction_data = None
    elif options.arg_prediction_type == 'cnn_arg_predictor':
        arg_predictor = arg_predictors.cnn_arg_predictor
        prediction_data = predict_arg_order_all_tasks_with_prob(examples)
    elif options.arg_prediction_type == \
            'mix_fixation_provided_and_cnn_arg_predictor':
        arg_predictor = arg_predictors.mix_fixation_provided_and_cnn_arg_predictor
        prediction_data = predict_arg_order_all_tasks_with_prob(examples)
    else:
        raise ValueError("Invalid input to arg_prediction_type")

    start_time = time.time()
    program_search_example(options.n_programs_to_search,
                           all_programs,
                           learnt_progs,
                           examples,
                           prediction_data,
                           options.order,
                           arg_predictor,
                           concept_emulator,
                           explore_mode=options.explore_mode,
                           debug=True)
    print("time taken {}".format(time.time() - start_time))
