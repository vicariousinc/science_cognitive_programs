from itertools import product, chain

import numpy as np

from cognitive_programs.program_induction.program_compression import \
    expand_to_atoms, sat_to_sub_and_starts


class CondProgramModel(object):
    def __init__(self, program_model, arg_log_p, valid_atoms):
        log_T = program_model.log_T.copy()

        # Inject in log_T the knowledge that some instructions/subroutines will never appear
        n_instr = log_T.shape[0]
        valid = np.zeros(n_instr, dtype=bool)
        for instr in xrange(n_instr):
            valid[instr] = True
            if instr == program_model.eop_marker:
                valid[instr] = False
            else:
                for atom in expand_to_atoms(np.array([instr]), program_model.sub_info):
                    if atom not in valid_atoms:
                        valid[instr] = False
        log_T[:, ~valid] = -np.inf

        # Augment with the probability of each parameterization
        n_instr = log_T.shape[0]
        self.interpretations = []
        new_log_T = []
        vertical_expansion = []
        for instr in xrange(n_instr):
            if instr == program_model.eop_marker:
                prog_atoms = (program_model.eop_marker,)
                arg_choices = [[('EOParg', -np.inf)]]
            else:
                prog_atoms = expand_to_atoms(np.array([instr]), program_model.sub_info)
                arg_choices = [arg_log_p[i] for i in prog_atoms]
            log_ps = []
            for arg_choice in product(*arg_choices):
                args, log_p = zip(*arg_choice)
                log_p = sum(log_p)
                self.interpretations.append(tuple(zip(prog_atoms, args)))
                vertical_expansion.append(instr)
                log_ps.append(log_p)
            new_log_T.append(log_T[:, instr:instr + 1] + np.array(log_ps).reshape(1, -1))
        new_log_T = np.hstack(new_log_T)
        self.log_T = np.vstack([new_log_T[row, :] for row in vertical_expansion])
        self.initial_state = vertical_expansion.index(program_model.eop_marker)

        sat = [zip(*s)[0] for s in self.interpretations]
        self.sub_info = sat_to_sub_and_starts(sat) + (0,)

    def expand_to_program(self, prog):
        return tuple(chain(*[self.interpretations[i] for i in prog]))

    def expand_to_program_broken_by_subroutines(self, prog):
        return tuple([self.interpretations[i] for i in prog])

    def program_length_and_code(self, prog_broken_by_subr):
        """Example usage when the input contains nested subroutine:
        prog_broken_by_subr = [(0, None), (6, 'y'), (25, None), ((1, None),
        (2, None), (3, None)), (10, None), (8, None), (12, None), (14, None),
        (26, None),(16,None),(6,'g'), ((1, None), (2, None), (3, None)),
        (10, None), (8, None)]
        cond_model.program_length_and_code(prog_broken_by_subr)"""
        code = []
        for instr in prog_broken_by_subr:
            if isinstance(instr[0], int):
                instr = (instr,)
            code.append(self.interpretations.index(instr))
            if instr not in self.interpretations:
                return np.inf
        prog_idx = np.vstack((np.hstack((self.initial_state, code[:-1])).astype(int), code))
        print -self.log_T[prog_idx[0, :], prog_idx[1, :]]
        return -self.log_T[prog_idx[0, :], prog_idx[1, :]].sum(), code
