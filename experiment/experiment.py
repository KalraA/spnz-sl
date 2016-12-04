import numpy as np

class Experiment:
        def __init__(self, model, trainfiles, testfiles):
                self.model = model
                self.trainfiles = trainfiles
                self.testfiles = testfiles

        def train_one_file(self, filename, upd):
                dtype = int if self.model.params.binary else float
                obs = np.loadtxt(filename, delimiter=",", dtype=dtype)
                self.model.update(obs, upd)

        def train(self):
                i = 0
                for filename in self.trainfiles:
                        print(filename)
                        i += 1
                        if i == 1:
                                self.train_one_file(filename, True)
                        else:
                                self.train_one_file(filename, False)

        def evaluate_one_file(self, filename):
                dtype = int if self.model.params.binary else float
                obs = np.loadtxt(filename, delimiter=",", dtype=dtype)
                logprob = 0
                n = 100000
                for i in range((len(obs)-1)//n + 1):
                        a = i*n
                        b = min(len(obs), a + n)
                        logprob += np.sum(self.model.evaluate(obs[a:b]))
                logprob = logprob/float(len(obs))
                #logprob = self.model.evaluate(obs)
                return logprob

        def evaluate(self):
                logprob_total = 0.0
                n_total = 0
                for filename in self.testfiles:
                        logprob = self.evaluate_one_file(filename)
                        # logprob_total += np.sum(logprob)
                        # n_total += len(logprob)
                return logprob #_total / n_total

        def run(self):
                self.train()
                result = self.evaluate()
                return result
