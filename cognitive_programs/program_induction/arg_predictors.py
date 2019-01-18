import numpy as np
from cognitive_programs.program_induction.utils import create_commands_dict
from scipy.special import logsumexp
from cognitive_programs.arg_prediction.eval_classifier_cnn import PROB_THRESHOLD


def uniform_arg_predictor(concept, prediction_data, all_programs):
    command_dict = create_commands_dict()
    valid_atoms = list(command_dict.keys())
    max_instr = max(valid_atoms)
    arg_log_p = {}
    for instr in xrange(max_instr + 1):
        args = command_dict.get(instr, (None, None))[1]
        if args is None:  # correct for a None outside of list
            args = [None]
        log_p = -np.log(len(args))
        arg_log_p[instr] = [(arg, log_p) for arg in args]
    return arg_log_p, valid_atoms


def ground_truth_arg_predictor(concept, prediction_data, all_programs):
    program = all_programs[concept]
    command_dict = create_commands_dict()
    command_set = set(command_dict.keys())
    max_instr = max(command_set)
    arg_log_p = {}
    for instr in xrange(max_instr + 1):
        args = set([arg for i, arg in program if i == instr])
        if not args:
            args = [None]
        log_p = -np.log(len(args))
        arg_log_p[instr] = [(arg, log_p) for arg in args]

    prog_instrs = zip(*program)[0]
    valid_atoms = []

    for instr in command_set:
        if instr in prog_instrs or command_dict[instr][1] is None:
            valid_atoms.append(instr)
    return arg_log_p, valid_atoms


def cnn_arg_predictor(concept,
                      prediction_data,
                      all_programs,
                      all_or_nothing_probs=True):
    arg_p = prediction_data[concept]
    command_dict = create_commands_dict()
    valid_atoms = list(command_dict.keys())
    max_instr = max(valid_atoms)
    arg_log_p = {}
    for instr in xrange(max_instr + 1):
        if instr in arg_p:
            # this value should not contain all -inf, threshold if that happens
            log_p = np.log(zip(*arg_p[instr])[1])
            if (log_p < np.log(PROB_THRESHOLD)).all():
                valid_atoms.remove(instr)
            if all_or_nothing_probs:
                log_p[log_p < np.log(PROB_THRESHOLD)] = -1e10
                log_p /= 100
            log_p -= logsumexp(log_p)
            arg_log_p[instr] = zip(zip(*arg_p[instr])[0], log_p)
        elif instr not in valid_atoms:
            arg_log_p[instr] = [(None, 0.0)]
        else:
            args = command_dict[instr][1]
            if args is None:  # correct for a None outside of list
                args = [None]
            log_p = -np.log(len(args))
            arg_log_p[instr] = [(arg, log_p) for arg in args]
    return arg_log_p, valid_atoms


def mix_fixation_provided_and_cnn_arg_predictor(concept,
                                                prediction_data,
                                                all_programs,
                                                all_or_nothing_probs=True):
    """ In addition to predicting the args using CNN, this function also provides
    fixation guidence"""
    arg_log_p, valid_atoms = cnn_arg_predictor(concept, prediction_data,
                                               all_programs,
                                               all_or_nothing_probs=all_or_nothing_probs)

    program = all_programs[concept]

    instr = 11
    args = set([arg for i, arg in program if i == instr])
    if not args:
        args = [None]
        valid_atoms.remove(instr)
    log_p = -np.log(len(args))
    arg_log_p[instr] = [(arg, log_p) for arg in args]

    return arg_log_p, valid_atoms
