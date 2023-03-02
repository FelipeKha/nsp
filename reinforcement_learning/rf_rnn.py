import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import os

from utils.covering_cost import CoveringCost
from utils.get_population import GetPopulation

class ReingforcementLearningRNN:
    def __init__(
            self, 
            nb_nurses: int,
            nb_work_days_per_week: int,
            nb_shifts_per_work_day: int,
            nb_nrs_per_shift: int,
            nrs_max_work_days_per_week: int,
            H: int,
            batch_size: int,
            learning_rate: float,
            gamma: float,
            decay_rate: float,
            resume: bool,
            get_population: GetPopulation,
            covering_cost: CoveringCost,
    ) -> None:
        # assign instance variables
        self.nb_nurses = nb_nurses
        self.nb_work_days_per_week = nb_work_days_per_week
        self.nb_shifts_per_work_day = nb_shifts_per_work_day
        self.nb_nrs_per_shift = nb_nrs_per_shift
        self.nrs_max_work_days_per_week = nrs_max_work_days_per_week
        self.H = H
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.resume = resume
        self.get_population = get_population
        self.covering_cost = covering_cost
        
        work_dir_path = '/Users/felipekharaba/Documents/Documents – Felipe’s MacBook Pro/Coding courses/Projects/nurse_scheduling_problem/reinforcement_learning/'
        self.model_file_path = os.path.join(work_dir_path, 'model.pkl')

    def initialize_model(self) -> tuple(dict, dict, dict):
        D = self.nb_nurses \
            * self.nb_work_days_per_week \
            * self.nb_shifts_per_work_day
        if self.resume:
            with open(self.model_file_path, 'rb') as f:
                model = pickle.load(f)
        else:
            model = {}
            model['W1'] = np.random.randn(self.H, D) / np.sqrt(D)
            model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)
        grad_buffer = { k : np.zeros_like(v) for k,v in iter(model.items()) }
        rmsprop_cache = { k : np.zeros_like(v) for k,v in iter(model.items()) }
        return model, grad_buffer, rmsprop_cache

    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))
    
    def discount_rewards(self, r: list) -> list:
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            # if r[t] != 0: running_add = 0 # not sure if this is necessary
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    def policy_forward(
            self, 
            x: np.ndarray, 
            model: dict
        ) -> tuple(float, float):
        h = np.dot(model['W1'], x)
        h[h<0] = 0
        logp = np.dot(model['W2'], h)
        p = self.sigmoid(logp)
        return p, h
    
    def policy_backward(
            self, 
            epx: np.ndarray, 
            eph: np.ndarray, 
            epdlogp: np.ndarray, 
            model: dict,
        ) -> dict(str, np.ndarray):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, model['W2'])
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    def train_model(self):
        pass

    def search_solution(self):
        pass


# proposed by Copilot    
        # initialize weights
        self.hidden_layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, output_size, lambda x: x)
        self.hidden_layers.append(layer)

        # collect params for gradient descent
        self.params = []
        for layer in self.hidden_layers:
            self.params += layer.params

        # for momentum
        self.dparams = [np.zeros_like(p) for p in self.params]

    def predict(self, X):
        # forward propagation
        out = X
        for layer in self.hidden_layers:
            out = layer.forward(out)
        return out

    def sgd(self, X, Y, print_period=10, max_iter=100):
        # keep track of the losses and the number of examples seen for each epoch
        losses = []
        N = len(Y)
        num_examples_seen = 0

        for epoch in range(max_iter):
            # randomly shuffle the data
            permutation = np.random.permutation(N)
            X = X[permutation]
            Y = Y[permutation]

            # perform stochastic gradient descent
            for i in range(N):
                # do one step of gradient descent
                loss = self.train(X[i], Y[i])
                losses.append(loss)

                # print the loss
                num_examples_seen += 1
                if print_period > 0 and num_examples_seen % print_period == 0:
                    print("Loss after num_examples_seen=%d epoch=%d: %f" % (num_examples_seen, epoch, loss))

        return losses

    def train(self, x, y):
        # forward propagation
        out = x





