import numpy as np
from cognitive_programs.program_induction.subroutine_discovery import find_all_subseqs_naive, validate_sub
from scipy.special import logsumexp
import numba as nb


def sat_to_sub_and_starts(sat):
    sub_ends = np.array([len(sub) for sub in sat])
    sub_ends = np.cumsum(sub_ends)
    subs = np.array([instr for sub in sat for instr in sub])
    return subs, sub_ends


@nb.njit
def expand_to_atoms(program, sub_info):
    # sub_info = subs, sub_ends, first_subroutine
    # if first_subroutine = 0, only one level of expansion
    # default argument for sub_info: (np.array([], dtype=int), np.array([], dtype=int), 1000000000)
    subs, sub_ends, first_subroutine = sub_info
    assert (0 <= subs).all() and (subs < first_subroutine + len(subs)).all(), "Some subroutine points outside of range"
    assert (0 <= program).all() and (program < first_subroutine + len(subs)).all(), "Program points outside of range"
    assert first_subroutine == 0 or (program < first_subroutine).all() or (sub_ends[0] > 1 and (np.diff(sub_ends) > 1).all())

    prog_as_atoms = [-1]
    for instr in program:
        if instr < first_subroutine:
            prog_as_atoms.append(instr)
        else:
            if first_subroutine == 0:  # all instr are subroutines, only one level of expansion
                sub_idx = instr
                sub_start = sub_ends[sub_idx - 1] if sub_idx > 0 else 0
                sub_end = sub_ends[sub_idx]
                prog_as_atoms.extend(subs[sub_start:sub_end])
            else:
                sub_idx = instr - first_subroutine
                sub_start = sub_ends[sub_idx - 1] if sub_idx > 0 else 0
                sub_end = sub_ends[sub_idx]
                subr_as_atoms = expand_to_atoms(subs[sub_start:sub_end], sub_info)
                prog_as_atoms.extend(subr_as_atoms)
    return prog_as_atoms[1:]


def encode(array):
    """ Array to unicode string """
    return ''.join(map(unichr, array))


def decode(string):
    """ Unicode string to array"""
    return np.array(map(ord, string), dtype=int)


def array_to_tuples(array, eop_marker, remove_last=True):
    eop_locations = (array == eop_marker).nonzero()[0]
    start = 0
    lot = []
    for eop_loc in eop_locations:
        if remove_last:
            lot.append(tuple(array[start:eop_loc].tolist()))
        else:
            lot.append(tuple(array[start:eop_loc + 1].tolist()))
        start = eop_loc + 1
    return lot


class ProgramModel(object):
    def __init__(self, programs, n_atoms, open_instr, close_instr, pseudo_count=1e-4,
                 no_pseudo_count_from=[], no_pseudo_count_to=[], add_eop_marker=True):
        """ Model for a list of programs.
            The programs are stored initially as a collection of atomic instructions.
            Method `insert_subroutine` allows inserting a subroutine in the model (and replaces it wherever it is present,
            both programs and other previously existing subroutines). This updates the representation of the collection of programs.
            Method `size` computes the size of a given program representation.
            When the method `compress` is called, it will greedily inser the next best subroutine in terms of compression.
            All potential subroutines are exhaustively tested
        Parameters
        ----------
        programs : [(p1_instr_1, ..., p1_instr_a), ..., (pn_instr_1, ..., pn_instr_b)]
            List of programs. Each program is a tuple of atomic instructions.
        n_atoms : int
            Number of atomic instructions. This should probably be the largest used instruction + 1. Or + 2 if the instructions
            do not include an End Of Program marker.
        open_instr : list
        close_instr : list
        pseudo_count : float
            Pseudocount to use when computing the frecuencies of instruction coocurrences (to estimate the transition matrix)
        add_eop_marker : Boolean, optional
            Whether the provided programs are all already finished with an end-of-program marker
            (the default is True, which will add EOP markers to the programs)
        """
        self.n_atoms = n_atoms
        self.log_T = np.zeros((n_atoms, n_atoms)) - np.log(n_atoms)  # log transition matrix, all rows must sum to one
        self.subroutines = np.array([], dtype=int)  # dictionary of subroutines
        self.eop_marker = self.n_atoms - 1
        self.open_instr = np.array(open_instr)
        self.close_instr = np.array(close_instr)
        self.n_subroutines = 0
        self.pseudo_count = pseudo_count
        self.no_pseudo_count_from = no_pseudo_count_from
        self.no_pseudo_count_to = no_pseudo_count_to

        # Convert to a single array and add EOP marker if necessary
        max_instr = 0
        program_list = []
        for i in xrange(len(programs)):
            max_instr = max(max_instr, max(programs[i]))
            program_list.extend(programs[i])
            if add_eop_marker:
                assert self.eop_marker not in programs[i], "Your programs have EOP markers and yet you're asking to add them..."
                program_list.append(self.eop_marker)
            assert program_list[-1] == self.eop_marker, "Last instruction of every program should be %d (end-of-program marker)" % self.eop_marker

        assert max_instr < self.n_atoms, "Some of the programs instructions are larger than %d, which you stated to be your largest atomic instruction" % (self.n_atoms - 1)
        if add_eop_marker and (max_instr != self.n_atoms - 2):
            print "Warning: You have established the number of non-EOP atoms to %d, but the largest used instruction is %d" % (self.n_atoms - 1, max_instr)
        if not add_eop_marker and (max_instr != self.n_atoms - 1):
            print "Warning: You have established the number of atoms to %d, but the largest used instruction is %d" % (self.n_atoms, max_instr)
        self.programs = np.array(program_list, dtype=int)
        assert len(self.programs) > 0, "No input progams were found"

    def copy(self):
        new_object = self.__class__([[0]], 2, [], [])
        new_object.n_atoms = self.n_atoms
        new_object.log_T = self.log_T.copy()
        new_object.subroutines = self.subroutines.copy()
        new_object.programs = self.programs.copy()
        new_object.eop_marker = self.eop_marker
        new_object.open_instr = self.open_instr.copy()
        new_object.close_instr = self.close_instr.copy()
        new_object.n_subroutines = self.n_subroutines
        new_object.pseudo_count = self.pseudo_count
        new_object.no_pseudo_count_from = self.no_pseudo_count_from
        new_object.no_pseudo_count_to = self.no_pseudo_count_to
        return new_object

    def adjust_parameters(self, order):
        """Adjust self.p_atoms and self.is_p_instr_is_atom based on program data
        """
        if order == 0:
            if len(self.subroutines) > 0:
                assert self.subroutines.max() < self.n_atoms + self.n_subroutines
            assert self.programs.max() < self.n_atoms + self.n_subroutines
            count_all = self.pseudo_count + np.zeros(self.n_atoms + self.n_subroutines)
            count_all[self.no_pseudo_count_to] = 0

            idx, counts = np.unique(self.subroutines, return_counts=True)
            count_all[idx] += counts
            idx, counts = np.unique(self.programs, return_counts=True)
            count_all[idx] += counts

            np.seterr(divide='ignore')
            logcount = np.log(count_all)
            np.seterr(divide='warn')
            self.log_T = np.empty((self.n_atoms + self.n_subroutines, self.n_atoms + self.n_subroutines))
            self.log_T[:] = logcount - logsumexp(logcount, keepdims=True)

        elif order == 1:
            count_all = self.pseudo_count + np.zeros((self.n_atoms + self.n_subroutines, self.n_atoms + self.n_subroutines))
            count_all[self.no_pseudo_count_from, :] = 0
            count_all[:, self.no_pseudo_count_to] = 0
            np.fill_diagonal(count_all, 0)  # don't add self-transitions in the pseudo-count

            assert self.programs.max() < self.n_atoms + self.n_subroutines
            idx, counts = np.unique(np.vstack((np.hstack((self.eop_marker, self.programs[:-1])), self.programs)), axis=1, return_counts=True)
            count_all[idx[0, :], idx[1, :]] += counts

            if len(self.subroutines) > 0:
                assert self.subroutines.max() < self.n_atoms + self.n_subroutines
                idx, counts = np.unique(np.vstack((np.hstack((self.eop_marker, self.subroutines[:-1])), self.subroutines)), axis=1, return_counts=True)
                count_all[idx[0, :], idx[1, :]] += counts
            np.seterr(divide='ignore')
            logcount = np.log(count_all)
            np.seterr(divide='warn')
            self.log_T = logcount - logsumexp(logcount, axis=1, keepdims=True)
        else:
            assert False, "Unsupported order"

    def log_probability(self):
        prog_idx = np.vstack((np.hstack((self.eop_marker, self.programs[:-1])), self.programs))
        log_p = self.log_T[prog_idx[0, :], prog_idx[1, :]].sum()
        if len(self.subroutines) > 0:
            subr_idx = np.vstack((np.hstack((self.eop_marker, self.subroutines[:-1])), self.subroutines))
            log_p += self.log_T[subr_idx[0, :], subr_idx[1, :]].sum()
        return log_p

    def size(self):
        """ Compute the total (possibly compressed) size of all programs,
            including the subroutines dictionary needed for decompression
        """
        return -self.log_probability() / np.log(2)

    def insert_subroutine(self, subroutine):
        """ Insert the subroutine in the dictionary
        Parameters
        ----------
        subroutine : (p1_instr_1, ..., p1_instr_a)
            List of atomic or non-atomic instructions
        """
        for instr in subroutine:
            assert instr < self.n_atoms + self.n_subroutines, "You provided a subroutine that uses unknown instructions"
        if subroutine[-1] == self.eop_marker:  # remove EOP marker if it has it
            subroutine = subroutine[:-1]
        subr = encode(subroutine)
        assert unichr(self.eop_marker) not in subr, "You're inserting a subroutine that has an EOP marker that's not at the End Of Program"

        # Find and replace subroutine in existing programs
        subr_c = unichr(self.n_atoms + self.n_subroutines)

        programs_str = encode(self.programs)
        while subr in programs_str:
            programs_str = programs_str.replace(subr, subr_c)
        self.programs = decode(programs_str)

        subroutines_str = encode(self.subroutines)
        while subr in subroutines_str:
            subroutines_str = subroutines_str.replace(subr, subr_c)
        self.subroutines = decode(subroutines_str)

        self.subroutines = np.hstack((self.subroutines, subroutine, self.eop_marker))

        self.n_subroutines += 1
        assert self.n_atoms + self.n_subroutines < 2**16 - 1  # make sure we don't have too many total instructions for unicode to deal with it

    def programs_as_tuples(self, removeEOP=True):
        return array_to_tuples(self.programs, self.eop_marker, removeEOP)

    def subroutines_as_tuples(self, removeEOP=True):
        return array_to_tuples(self.subroutines, self.eop_marker, removeEOP)

    def compress(self, order=1, max_subroutines=100, debug=False):
        if debug:
            print "Initial parameters:", str(self.size()), "bits"
        self.adjust_parameters(order)
        if debug:
            print "Adjusted parameters:", str(self.size()), "bits"

        for sub_idx in xrange(max_subroutines):
            # find potential subroutines
            reps = find_all_subseqs_naive(self.subroutines_as_tuples() + self.programs_as_tuples(), 2)
            subs = [(list(sub[0]), sub[1]) for sub in reps if validate_sub(sub[0], self.open_instr, self.close_instr)]

            # evaluate each subroutine
            min_size = self.size()
            best_sub = None
            for sub, n_times in subs:
                new_model = self.copy()
                new_model.insert_subroutine(sub)
                new_model.adjust_parameters(order)
                if new_model.size() < min_size:
                    best_sub = sub
                    min_size = new_model.size()
            if best_sub is None:
                break

            if debug:
                print "Best subroutine", best_sub, "with size", min_size, "bits"

            # insert best subroutine
            self.insert_subroutine(best_sub)
            self.adjust_parameters(order)
            if debug:
                print "New size:", self.size(), "bits"
        self.subroutines_str = [encode(s) for s in self.subroutines_as_tuples(removeEOP=True)]
        self.create_sub_info()

    def expand_to_atoms(self, program):
        """ Insert the subroutine in the dictionary
        Parameters
        ----------
        subroutine : (p1_instr_1, ..., p1_instr_a)
            List of atomic or non-atomic instructions
        """
        for instr in program:
            assert instr < self.n_atoms + self.n_subroutines, "You provided a program that uses unknown instructions"
        if program[-1] == self.eop_marker:  # remove EOP marker if it has it
            program = program[:-1]
        prog_str = encode(program)
        assert unichr(self.eop_marker) not in prog_str, "You're expanding a program that has an EOP marker that's not at the End Of Program"

        # Find and replace subroutine in existing programs
        done = False
        while not done:
            for subr_idx in xrange(self.n_subroutines - 1, - 1, -1):
                subr_c = unichr(self.n_atoms + subr_idx)
                while subr_c in prog_str:
                    prog_str = prog_str.replace(subr_c, self.subroutines_str[subr_idx])
            done = True
            for instr_c in prog_str:
                if ord(instr_c) >= self.n_atoms:
                    done = False
                    break

        return tuple(decode(prog_str).tolist())

    def program_lengths(self):
        for p in self.programs_as_tuples():
            prog_idx = np.vstack((np.hstack((self.eop_marker, p[:-1])).astype(int), p))
            yield -self.log_T[prog_idx[0, :], prog_idx[1, :]].sum()

    def create_sub_info(self):
        sat = self.subroutines_as_tuples()
        if len(sat) > 0:
            self.sub_info = sat_to_sub_and_starts(sat) + (self.n_atoms,)
        else:
            self.sub_info = (np.array([], dtype=int), np.array([], dtype=int), 1000000000)
