import time
import numpy as np
import numpy.random as npr
import copy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


###################
# MARK: CONSTANTS #
###################


# CONFIGURATION CONSTANTS
# These are the ones you're going to want to mess around with

MAX_DEPTH     = 50   # Number of layers possible
MAX_HEIGHT    = 50   # Maximum amount of nodes per layer
NUM_FOLDS     = 4    # Number of trials that should be run
EPOCHS        = 10   # Depth of training in each trial 
BATCH_SIZE    = 1000 # Training examples per epoch

POPULATION_SIZE = 30
NUM_GEN         = 20

P_CROSS = 5.0 / float(POPULATION_SIZE)

P_MUT = 5.0 / float(POPULATION_SIZE)
P_MUT_LAYERS = 0.2
P_MUT_WIDTHS = 0.2
P_MUT_ACTIVATIONS = 0.2
P_MUT_LEARNING = 0.1


# DATASET CONSTANTS
# Don't need to mess with these 

ACTIVATIONS   = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER     = "adam"
METRICS       = ["accuracy"]
VERBOSE       = 0    # 0 for no output, 1 for extensive, 2 for some other option

(X, y), (_, _) = mnist.load_data()
X = np.reshape(X, (60000, 28 * 28, 1))


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
        if active_layers is None:
            self.active_layers = (npr.randint(0, 2, MAX_DEPTH) == 1)

        # Take the given layer widths, otherwise generate random ones
        self.layer_widths = layer_widths
        if self.layer_widths is None:
            self.layer_widths = npr.randint(2, MAX_HEIGHT, MAX_DEPTH)

        # Take the given activations, otherwise generate random ones from the
        # list provided above
        self.activations = activations
        if self.activations is None:
            self.activations = np.array([npr.choice(ACTIVATIONS) for _ in
                range(MAX_DEPTH)])

        # Take the given learning rate, otherwise generate a random one between
        # 0 and 1
        self.alpha = alpha
        if self.alpha is None:
            self.alpha = npr.random_sample([1]) / 10.0

        self.calculate_fitness()

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

    def calculate_fitness(self):
        """ Tests the fitted model against a dataset and generates the relevant
        metrics
        """
        if self.fitness is not None:
            self.__build__()
            kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
            
            # Find overall accuracy and the elapsed time
            start = time.time()
            self.results = cross_val_score(self.model, X, y, cv=kfold, verbose=VERBOSE,
                    error_score='raise')
            end = time.time()
            self.fitness = self.results.mean() # / (end - start)
        

######################
# MARK: GENETIC SHIZ #
######################


def initialize_population():
    return [NeuralNetwork(28 * 28, 10) for _ in range(POPULATION_SIZE)]

def select_parents(networks):
    gen = []

    fitness_reduce = lambda x: x.fitness
    # Generate the list of tournament selected parents
    for _ in range(POPULATION_SIZE):
        parents = np.random.choice(networks, 3, replace=False)
        selected_idx = np.argmax(np.vectorize(fitness_reduce)(parents))
        gen.append(parents[selected_idx])

    return np.array(gen)

def crossover(network1, network2):

    if npr.random_sample() >= P_CROSS:
        return (network1, network2)

    n1_lay = network1.active_layers
    n2_lay = network2.active_layers

    # depth
    for i in range(MAX_DEPTH):
        a = npr.randint(0, 2)
        if a == 1:
            temp = n1_lay[i]
            n1_lay[i] = n2_lay[i]
            n2_lay[i] = temp

    # width
    cp = npr.randint(0, MAX_HEIGHT)
    n1_wid = np.append(network1.layer_widths[:cp], network2.layer_widths[cp:])
    n2_wid = np.append(network2.layer_widths[:cp], network1.layer_widths[cp:])

    # activation

    cp = npr.randint(0, len(ACTIVATIONS))
    n1_act = np.append(network1.activations[:cp], network2.activations[cp:])
    n2_act = np.append(network2.activations[:cp], network1.activations[cp:])

    # alpha

    n1_alpha = (network1.alpha + network2.alpha) * .5
    n2_alpha = (network1.alpha + network2.alpha) * .5

    n1 = NeuralNetwork(28 * 28, 10, 
                       active_layers=n1_lay, 
                       layer_widths=n1_wid, 
                       activations=n1_act, 
                       alpha=n1_alpha)
    n2 = NeuralNetwork(28 * 28, 10, 
                       active_layers=n2_lay, 
                       layer_widths=n2_wid, 
                       activations=n2_act, 
                       alpha=n1_alpha)

    return n1, n2

def mutate(network):

    ntwk = copy.deepcopy(network)

    #20 percent chance of bit flipping
    if npr.random_sample() < P_MUT_LAYERS:
        #3 bits get possibly flipped
        for x in range(3):
            ntwk.active_layers[npr.randint(0,MAX_DEPTH)] = (npr.randint(0, 2) == 1)

    #20 percent chance of mutation in layer widths
    if npr.random_sample() < P_MUT_WIDTHS:
        for x in range(3):
            upDown = npr.randint(0,2)
            if upDown == 1:
                rInt = npr.randint(0,MAX_DEPTH)
                if ntwk.layer_widths[rInt] + 2 < MAX_DEPTH:
                    ntwk.layer_widths[rInt] += npr.randint(0, 2)
                else :
                    ntwk.layer_widths[rInt] -= npr.randint(0, 2)
            else:
                rInt = npr.randint(0,MAX_DEPTH)
                if ntwk.layer_widths[rInt] - 2 < MAX_DEPTH:
                    ntwk.layer_widths[rInt] -= npr.randint(0, 2)
                else :
                    ntwk.layer_widths[rInt] += npr.randint(0, 2)

    #20 percent chance of mutation in activation function
    if npr.random_sample() < P_MUT_ACTIVATIONS:
        for x in range(3):
            rInt = npr.randint(0,MAX_DEPTH)
            ntwk.activations[rInt] = npr.choice(ACTIVATIONS)

    #10 percent chance of learning rate being mutated, uniform mutation which will completely 
    #reassign learning rate so there is a smaller chance of this  
    if npr.random_sample() < P_MUT_LEARNING:
        ntwk.alpha = npr.random_sample([1]) / 10.0

    return network

def next_generation(networks):

    parents = select_parents(networks)
    survivors = []

    # Run crossover / mutation on survivors
    for i in range(int(POPULATION_SIZE / 2)):
        (c1, c2) = crossover(parents[i], parents[i + 1])
        survivors.append(mutate(c1))
        survivors.append(mutate(c2))

    return np.array(survivors)

def print_statistics(population):
    fitness_reduce = np.vectorize(lambda x: x.fitness)
    population_fitnesses = fitness_reduce(population)
    print("{:.2} & {:.2} & {:.2} & {:.2} \\\\ \\hline".format(
            np.min(population_fitnesses), 
            np.mean(population_fitnesses),
            np.max(population_fitnesses),
            np.std(population_fitnesses),
            )
        )
    

#################
# MARK: TESTING #
#################

print("\\begin{tabular}{|c|c|c|c|}")
print("\\hline Min & Average & Max & STD \\\\ \\hline")

population = initialize_population()
print_statistics(population)
# Do whatever the hell you want here
for _ in range(NUM_GEN):
    population = next_generation(population)
    print_statistics(population)

print("\\end{tabular}")

