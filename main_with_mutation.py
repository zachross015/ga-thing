import time
import numpy as np
import numpy.random as npr
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


###################
# MARK: CONSTANTS #
###################


ACTIVATIONS   = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER     = "adam"
METRICS       = ["accuracy"]

MAX_DEPTH     = 50   # Number of layers possible
MAX_HEIGHT    = 50   # Maximum amount of nodes per layer
NUM_FOLDS     = 2    # Number of trials that should be run
EPOCHS        = 10   # Depth of training in each trial 
BATCH_SIZE    = 1000 # Training examples per epoch
VERBOSE       = 0    # 0 for no output, 1 for extensive, 2 for some other option

POPULATION_SIZE = 10
NUM_GEN         = 0  # Definitely change this at some point lol


#######################
# MARK: NN DEFINITION #
#######################


class NeuralNetwork:

    def __init__(self, input_dim, output_dim, active_layers=None, layer_widths=None, activations=None, alpha=None):
        """ Intializes a neural network with the given components. Only required
        input is the input dimensions, the rest can be either random or passed
        on from a parent
        """

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Take the given active layers, otherwise generate random ones
        self.active_layers = active_layers
        if active_layers == None:
            self.active_layers = (npr.randint(0, 2, MAX_DEPTH) == 1)

        # Take the given layer widths, otherwise generate random ones
        self.layer_widths = layer_widths
        if self.layer_widths == None:
            self.layer_widths = npr.randint(2, MAX_HEIGHT, MAX_DEPTH)

        # Take the given activations, otherwise generate random ones from the
        # list provided above
        self.activations = activations
        if self.activations == None:
            self.activations = np.array([npr.choice(ACTIVATIONS) for _ in
                range(MAX_DEPTH)])

        # Take the given learning rate, otherwise generate a random one between
        # 0 and 1
        self.alpha = alpha
        if self.alpha == None:
            self.alpha = npr.random_sample([1]) / 10.0

        self.fitness = None

    def __str__(self):
        string = "Neural Network with dimensions {} x {}\n".format(self.input_dim, self.output_dim)
        string += "Layers: {}\n".format(self.layer_widths[self.active_layers])
        string += "Activations: {}\n".format(self.activations[self.active_layers])
        string += "Learning Rate: {}\n".format(self.alpha)
        if self.fitness is not None:
            string += "Fitness: {}\n".format(self.fitness)
        else:
            string += "Fitness not evaluated.\n"
        return string

    def __compile__(self):
        """ Creates and compiles the neural network given the parameters the
        network was initalized with
        """

        model = Sequential()
        lw = list(self.layer_widths[self.active_layers])
        ac = list(self.activations[self.active_layers])
        model.add(Dense(lw[0], input_dim=self.input_dim, activation=ac[0]))
        for i in range(1, len(lw)):
            model.add(Dense(lw[i], activation=ac[i]))
        model.add(Dense(self.output_dim, activation='sigmoid'))
        model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER,
                metrics=METRICS)
        return model

    def __build__(self):
        """ Builds the model into a classifier given the parameters the network
        was initalized with
        """
        self.model = KerasClassifier(build_fn=self.__compile__, epochs=EPOCHS,
                batch_size=BATCH_SIZE, verbose=VERBOSE)

    def calculate_fitness(self, X, y):
        """ Tests the fitted model against a dataset and generates the relevant
        metrics
        """
        self.__build__()
        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
        
        # Find overall accuracy and the elapsed time
        start = time.time()
        results = cross_val_score(self.model, X, y, cv=kfold, verbose=VERBOSE,
                error_score='raise')
        end = time.time()
        self.fitness = results.mean() / (end - start)
        return self.fitness
        

######################
# MARK: GENETIC SHIZ #
######################


def initialize_population():
    return [NeuralNetwork(28 * 28, 10) for _ in range(POPULATION_SIZE)]

def select_parents(networks):
    pass

def crossover(network1, network2):
    pass

def mutate(network):
    #20 percent chance of bit flipping
    if npr.random_sample() < 0.2:
        #3 bits get possibly flipped
        for x in range(3):
            network.active_layers[npr.randint(0,MAX_DEPTH)] = (npr.randint(0, 2) == 1)

    #20 percent chance of mutation in layer widths
    if npr.random_sample() < 0.2:
        for x in range(3):
            upDown = npr.randint(0,2)
            if upDown == 1:
                rInt = npr.randint(0,MAX_DEPTH)
                if network.layer_widths[rInt] + 2 < MAX_DEPTH:
                    network.layer_widths[rInt] += npr.randint(0, 2)
                else :
                    network.layer_widths[rInt] -= npr.randint(0, 2)
            else:
                rInt = npr.randint(0,MAX_DEPTH)
                if network.layer_widths[rInt] - 2 < MAX_DEPTH:
                    network.layer_widths[rInt] -= npr.randint(0, 2)
                else :
                    network.layer_widths[rInt] += npr.randint(0, 2)

    #20 percent chance of mutation in activation function
    if npr.random_sample() < 0.2:
        for x in range(3):
            rInt = npr.randint(0,MAX_DEPTH)
            network.activations[rInt] = npr.choice(ACTIVATIONS)

    #10 percent chance of learning rate being mutated, uniform mutation which will completely 
    #reassign learning rate so there is a smaller chance of this  
    if npr.random_sample() < 0.1:
        network.alpha = npr.random_sample([1]) / 10.0


def next_generation(networks):
    pass


#################
# MARK: TESTING #
#################


# Initialize the data sets. X is the input data while y is the output.
(X, y), (_, _) = mnist.load_data()
X = np.reshape(X, (60000, 28 * 28, 1))

population = initialize_population()
# Do whatever the hell you want here


