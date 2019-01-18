import gzip
import cPickle as pickle


class Oracle(object):
    def __init__(self, concept, examples):
        self.concept = concept
        filename = 'oracle_cache/oracle_data_for_{}.pkl'.format(concept)
        with gzip.GzipFile(filename, 'rb') as file:
            self.saved_results = pickle.load(file)

    def check_program(self, program, n_examples_to_test):
        assert program in self.saved_results, "Oracle dictionary incomplete!"
        return self.saved_results[program]
