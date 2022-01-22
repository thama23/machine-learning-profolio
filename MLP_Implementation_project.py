
# # Multi-layer Perceptron Implementation Project

# ## Part I - Model of a perceptron layer
#
# Part I implements a layer of perceptrons which takes an input from the dataset and outputs the activation (1 or 0) of each perceptron in the layer. The number of perceptrons in the layer is randomly chosen.
#
# Sample output:
# Activation output for each node in the node layer:  [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
#
# In this example, there are 10 perceptrons in the perceptron layer. Perceptrons 1, 2, 4, 6, 8 and 9 are activated (=1) and the other perceptrons are not (=0).

# In[1]:


'''
Part 1
------
'''

import numpy as np
import pandas as pd
import random

class Nodelayer:
    '''
    Create a layer of perceptrons of any number, and compute output for each node.
    '''
    # loads and normalizes data
    def data_processing(self,filename):
        '''
        Load and normalize the dataset
        :param filename: input data file as .csv or .data
        :return: input data as normalized pandas dataframe
        '''
        # load data in filename, i.e. iris or diabetes datasets
        if filename.split('.')[1] == 'data':
            data = []
            # open file
            with open(filename) as infile:
                    for line in infile.read().splitlines():
                        tokens = line.split(",")
                        data.append(tokens)
            # converts data to pandas dataframe called df
            df = pd.DataFrame(data)
        # if .csv in filename, e.g. pima indians diabetes dataset
        elif filename.split('.')[1] == 'csv':
            # converts data to pandas dataframe called df
            df = pd.read_csv(datafile,header=None)
        # else print error statement if tile not .data or .csv
        else:
            print('File must end with .csv or .data extension.')

        # normalize data
        df_x = df.iloc[:,:-1].apply(pd.to_numeric).dropna() # convert features to numerical data
        # computes normalized numerical data
        compute_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
        # join with labels column
        df_norm = compute_norm.join(df.iloc[:,len(df.columns)-1])

        return df_norm

    # extracts single input vector from a dataset
    def data_instance(self,df_norm):
        '''
        Extracts input vector from a normalized dataset
        param df_norm: normalized dataset
        :return: randomly selected input vector from normalized dataset
        '''
        # randomly select data instance via dataset index
        instance_index = random.randint(0,df_norm.shape[0]-1) # minus 1 due to 0 indexing

        # extract input array
        input_vector = df_norm.iloc[instance_index,:-1].values

        return input_vector

    # create node layer structure
    def node_structure(self, input_vector, num_nodes):
        '''
        Implements a layer of nodes of any number
        :param input_vector: input vector
        :param num_nodes: number of nodes in the layer
        :return: output (1 or 0) for each neuron
        '''
        out = [] # list to store output (1 or 0) for each neuron in neuron layer
        for node in range(num_nodes):
            # how many weights to generate, i.e. length of weight vector
            num_weights = len(input_vector)
            # randomly generate weights, uniformly distributed from -1 to 1
            weight_vector = np.array([np.random.uniform(-1,1) for i in range(0,num_weights)])
            bias = np.random.uniform(-1,1) # set bias in a similar fashion

            # compute neuron activation
            activation_input = sum(weight_vector * input_vector) + bias

            # node output
            if activation_input >= 0:
                node_out = 1
            else:
                node_out = 0

            # store output for each neuron
            out.append(node_out)

        return out

# Driver for the program.
if __name__ == '__main__':
    print(' \n Part 1:\n','-' * len('Part 1:'))
    # create class
    node_layer = Nodelayer()
    # load data from file and normalize data
    df_norm = node_layer.data_processing('iris.data') # or pima indian diabetes dataset
    # retrieve input vector from normalized dataset
    input_vector = node_layer.data_instance(df_norm)
    # set arbitrary number (with a max of 10 in this case) of nodes in the node layer
    num_nodes = random.randint(1,10)
    # create neuron layer structure and compute output
    nodelayer_out = node_layer.node_structure(input_vector,num_nodes)
    # print activation output for each node
    print('Activation output for each node in the node layer: ', nodelayer_out)


# # Part II - MLP without back-propagation
#
# In Part II, a Multi-layer Perceptron (MLP) network is built with an arbitrary number of layers and nodes/perceptrons in each layer.
#
# Classifer sample output (iris dataset):
#
# Network structure: [5, 6, 2, 3];
# Total number of dataset instances tested:  150;
# Accuracy score: 34.0%
#
# The sample output from the statements above indicate that this network was randomly generated to have 4 layers (excluding the input layer), with 5 nodes in hidden layer 1, 6 nodes in hidden layer 2, 2 nodes in hidden layer 3, and 3 nodes in output layer (given iris dataset is a 3-class problem). After looping through all 150 instances
# of the dataset, the accuracy using this particular network and weights was found to be 34.0%. Given that the network weights are randomly generated, it is expected that the accuracy score hovers around 33.33% (1 answer randomly selected out of 3 possibilities).

# In[2]:


'''
Part 2
------
'''

def node(vector):
    '''
    Computes weights, bias, activation potential and output (sigmoid activation) for a node
    :param vector: input vector from previous layer
    :return: output activation for the node
    '''
    global weight_vector, bias
    # randomly generate weights, uniformly distributed from -1 to 1
    weight_vector = np.array([np.random.uniform(-1,1) for i in range(0,len(vector))])
    # randomly generate weights for each node, uniformly distributed from -1 to 1
    bias = np.random.uniform(-1,1)
    # compute activation potential
    activation_potential = sum(weight_vector * vector) + bias
    # use sigmoid activation function
    node_out = sigmoid(activation_potential)

    return node_out

def node_layer(num_nodes):
    '''
    Computes output of neuron layer
    :param num_nodes: input vector from previous layer
    :return: output activation for each node in the layer
    '''
    layer_out = [] # list to store activations of each node in node layer
    global num, vector
    # loop through each node in node layer
    for num in range(num_nodes):
        # compute activation output for each node
        node_out = node(vector)
        layer_out.append(node_out) # list containing output activations for each node
    # output of node layer is the input vector to the nodes of the next layer
    vector = layer_out

    return vector

def network(structure):
    '''
    Builds a network with an arbitrary number of layers and nodes in each layer
    :param structure: the network's architecture specified as a list
                      e.g. [2,4,3] is a network with 3 layers, 2 nodes in hidden layer 1,
                      4 nodes in hidden layer 2, and 3 nodes in the output layer
    :return: network output
    '''
    global num, i
    num_layers = len(structure) # number of layers given network structure set as a list
    for i in range(num_layers): # for each layer
        num_nodes = structure[i] # compute number of nodes in that layer
        vector = node_layer(num_nodes) # compute output of this layer

    return vector

def sigmoid(z):
    '''
    Sigmoid activation function
    :param z: activation potential
    :return: computes sigmoid activation function
    '''
    sig = 1/(1 + np.exp(-z))
    return sig

def predict(output_scores):
    '''
    Converts output of an node layer to class predictions
    :param output_scores: output of an node layer
    :return: predictions as a list, e.g. [0,1,0] if the second class if predicted out of 3
    '''
    global prediction
    prediction = []
    # loops through output scores
    for idx, val in enumerate(output_scores):
        max_index = np.argmax(output_scores) # finds the index of the highest score
        # sets max score to 1, rest to 0
        if idx != max_index:
            prediction.append(0)
        else:
            prediction.append(1)
    return prediction

# Driver for the program.
if __name__ == '__main__':
    print(' \n Part 2:\n','-' * len('Part 2:'))
    # load data, in this case iris dataset
    filename = 'iris.data' # or 'diabetes.csv' to load pima indian diabetes dataset
    if filename.split('.')[1] == 'data':
            data = []
            # open file
            with open(filename) as infile:
                    for line in infile.read().splitlines():
                        tokens = line.split(",")
                        data.append(tokens)
            # converts data to pandas dataframe called df
            df = pd.DataFrame(data)
        # if .csv in filename, e.g. pima indians diabetes dataset
    elif filename.split('.')[1] == 'csv':
        # converts data to pandas dataframe called df
        df = pd.read_csv(datafile,header=None)
    # else print error statement if tile not .data or .csv
    else:
        print('File must end with .csv or .data extension.')

    # normalize data
    df_x = df.iloc[:,:-1].apply(pd.to_numeric).dropna() # convert features to numerical data
    # computes normalized numerical data
    compute_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
    # join with labels column
    df_norm = compute_norm.join(df.iloc[:,len(df.columns)-1])

    # replace iris labels column with 3 prediction outputs: [1,0,0], [0,1,0], [0,0,1]
    df_norm['labels'] = df_norm.iloc[:,-1:] # add labels column
    global df_norm_pred
    df_norm_pred = df_norm.labels.map({'Iris-setosa':[1,0,0], # replace with predictions
                                   'Iris-versicolor':[0,1,0],
                                   'Iris-virginica':[0,0,1]}).to_frame()
    df_norm_pred = pd.concat([df_norm.iloc[:,:-2], df_norm_pred], axis=1) # remove extra col

    # generate network architecture with arbitrary number of layers and nodes in each layer
    n_layers = random.randint(2,5) # min layers=2, max layers = 5 (randomly relected)
    network_struc = random.sample(range(1, 10), n_layers) # generate random network structure
    # set output layer to have as many nodes as the problem has classes
    network_struc[-1] = len(df_norm.labels.unique()) # number of unique classes
    print('Network structure: ',network_struc)

    # Compute classification score
    corr_pred = 0 # counter to compute correct predictions
    incorr_pred = 0 # counter to compute incorrect predictions

    # loop through all data instances in the dataset
    for index, row in df_norm_pred.iterrows():
        # extract input vector
        vector = row[:-1].values

        # compute prediction
        pred = predict(network(network_struc)) # true prediction

        # correct classification count
        true_class = row[-1]# true class for that specific input vector

        # increment corr_pred if the prediction equals the true label, else increment incorr_pred
        if pred == true_class:
            corr_pred += 1
        else:
            incorr_pred += 1

    # accuracy score
    accuracy_score = corr_pred / (corr_pred + incorr_pred)
    print('Total number of dataset instances tested: ',corr_pred + incorr_pred)
    # accuracy should be circa 33% given random computer of 3 classes
    print('Accuracy score: {}{}'.format(round(accuracy_score * 100,2),'%'))


# # Part III - MLP with back-propagation
#
# In Part II, a MLP back-propagation network is created and used to classify the iris and pima indian diabetes datasets.
#
# To classify the iris data, the MLP has the following structure (including input layer): (4, 3, 3) , i.e. 4 nodes in the input layer (equals number of dataset features), 1 hidden layer with 3 nodes, and 3 nodes in output layer (one for each class in the problem)
#
# To classify the diabetes data, the MLP has the following structure (including input layer) (8, 4, 1):  , i.e. 8 nodes in the input layer (equals number of dataset features), 1 hidden layer with 4 nodes, and 1 node in output layer (binary classification problem)
#
# Sample output:
#
# Network shape: (4, 3, 3);
# Iris data classification score: 93.33%
#
# Network shape: (8, 4, 1);
# Diabetes data classification score: 71.24%
#
#

# In[3]:


'''
Part 3
------
'''

class MlpNetwork:
    """A back-propagation MLP network with sigmoid activation function"""

    # constructor
    def __init__(self, layer_size):
        """Initialize the network"""

        # initialize layer count, network shape and weights
        self.layer_count = 0
        self.network_shape = None
        self.weights = []

        # layer information (count and shape)
        self.layer_count = len(layer_size) - 1
        self.network_shape = layer_size

        # Data from last network run
        self._layer_input = []
        self._layer_output = []
        self._prev_weight_delta = []

        # Create the weight arrays
        for (l1,l2) in zip(layer_size[:-1], layer_size[1:]):
            self.weights.append(np.random.normal(scale=0.1, size = (l2, l1+1)))
            self._prev_weight_delta.append(np.zeros((l2, l1+1)))

    # Run the network
    def run_network(self, input_data):
        '''
        Run the network based on the input data
        :param input_data: input data
        :return: output of last layer of the network
        '''

        ln_cases = input_data.shape[0]

        # clear out the previous i/o values for the layers
        self._layer_input = []
        self._layer_output = []

        # run the network
        for i in range(self.layer_count):
            # Determine layer input
            if i == 0:
                layer_input = self.weights[0].dot(np.vstack([input_data.T, np.ones([1, ln_cases])]))
            else:
                layer_input = self.weights[i].dot(np.vstack([self._layer_output[-1], np.ones([1, ln_cases])]))

            self._layer_input.append(layer_input)
            self._layer_output.append(self.sgm(layer_input))

        return self._layer_output[-1].T

    # train the network
    def train_network(self, input_data, target, learning_rate = 0.2, momentum = 0.5):
        '''
        This method trains the network for one epoch
        :param input_data: input data
        :param target: target class
        :param learning_rate: learning rate of the algorithm
        :param momentum: momentum to not get stuck in local minima
        '''

        delta = []
        ln_cases = input_data.shape[0]

        # first run the network
        self.run_network(input_data)

        # calculate our deltas for backpropagation algorithm
        for i in reversed(range(self.layer_count)):
            if i == self.layer_count - 1:
                # Compare to the target values
                output_delta = self._layer_output[i] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.sgm(self._layer_input[i], True))
            else:
                # Compare to the following layer's delta
                delta_pullback = self.weights[i + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.sgm(self._layer_input[i], True))

        # Compute weight deltas
        for i in range(self.layer_count):
            delta_index = self.layer_count - 1 - i

            if i == 0:
                layer_output = np.vstack([input_data.T, np.ones([1, ln_cases])])
            else:
                layer_output = np.vstack([self._layer_output[i - 1], np.ones([1, self._layer_output[i - 1].shape[1]])])

            cur_weight_delta = np.sum(                                 layer_output[None,:,:].transpose(2, 0 ,1) * delta[delta_index][None,:,:].transpose(2, 1, 0)                                 , axis = 0)

            weight_delta = learning_rate * cur_weight_delta + momentum * self._prev_weight_delta[i]
            self.weights[i] -= weight_delta
            self._prev_weight_delta[i] = weight_delta

        return error

    # sigmoid activation function and its derivative
    def sgm(self, x, Derivative=False):
        '''
        Sigmoid function and its derivative for backpropagation weight updates
        :param x: input potential to apply sigmoid function or its detivative
        :param Derivative: whether or not to use the derivative of the sigmoid function
        :return: sigmoid function or its derivative, depending on use case
        '''
        if not Derivative:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            out = self.sgm(x)
            return out * (1.0 - out)

    def convert_out(self, out):
        '''
        Convert the output after running the network in order to perform classification
        :param out: output of the network
        :return: data formatted to compute classification score for iris dataset
        '''
        new_out = []
        for r,c in enumerate(out):
            for i,j in enumerate(c):
                if i == c.argmax():
                    new_out.append(1)
                else:
                    new_out.append(0)

        new_out = np.array([new_out[n:n+3] for n in range(0, len(new_out), 3)])

        return new_out

    def accuracy_score(self, model_pred, target_value):
        '''
        Custom function to compute accuracy
        :param model_pred: the network's predictions
        :param target_value: the true class labels
        :return: accuracy score as a percentage
        '''
        count = 0
        for i in range(model_pred.shape[0]):
            if (model_pred[i] == target_value[i]).all():
                count += 1
            else:
                pass

        accuracy_score = count/model_pred.shape[0] * 100
        return accuracy_score

# If run as a script, create a test object
if __name__ == "__main__":
    print(' \n Part 3:\n','-' * len('Part 3:'))
    # load iris data for classification
    data = []
    filename = 'iris.data'
    with open(filename) as infile:
            for line in infile.read().splitlines():
                tokens = line.split(",")
                data.append(tokens)
    # converts data to pandas dataframe called df
    df = pd.DataFrame(data)

    # normalize data
    df_x = df.iloc[:,:-1].apply(pd.to_numeric).dropna() # convert features to numerical data
    # computes normalized numerical data
    compute_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
    # join with labels column
    df_norm = compute_norm.join(df.iloc[:,len(df.columns)-1])

    # replace iris labels column with 3 prediction outputs: [1,0,0], [0,1,0], [0,0,1]
    df_norm['labels'] = df_norm.iloc[:,-1:] # add labels column

    df_norm_pred = df_norm.labels.map({'Iris-setosa':[1,0,0], # replace with predictions
                                   'Iris-versicolor':[0,1,0],
                                   'Iris-virginica':[0,0,1]}).to_frame()
    df_norm_pred = pd.concat([df_norm.iloc[:,:-2], df_norm_pred], axis=1) # remove extra col

    # split into training and test sets
    df_shuffled = df_norm_pred.sample(frac=1) # shuffle dataset
    test_data = df_shuffled.iloc[:int((1/5)*df_shuffled.shape[0]),:]
    train_data = df_shuffled.iloc[int((1/5)*df_shuffled.shape[0]):,:]
    test_data_x = np.array(test_data.iloc[:,:-1])
    test_data_y = test_data.iloc[:,-1:]
    train_data_x = np.array(train_data.iloc[:,:-1])
    train_data_y = train_data.iloc[:,-1:]

    # process data for training - both train and test labels
    train_target = []
    for i in train_data_y.values:
        train_target.append(i[0])
    train_target = np.array(train_target)

    test_target = []
    for i in test_data_y.values:
        test_target.append(i[0])
    test_target = np.array(test_target)

    # create network with structure (4,3,3)
    network_structure = (4,3,3)
    mlp = MlpNetwork(network_structure)

    # when to stop model training
    itermax = 100 # max iterations
    lmerr = 1e-6 # min error rate as stopping condition
    for i in range(itermax+1):
        err = mlp.train_network(train_data_x, train_target, momentum = 0.7)
        if err <= lmerr:
            print("Desired error reached. Iter: {0}".format(i))
            break

    # network output
    net_output = mlp.run_network(test_data_x)
    new_out = mlp.convert_out(net_output)

    # convert outputs to desired format for iris dataset
    accuracy = mlp.accuracy_score(new_out,test_target)

    # compute and print accuracy/ classification score
    print('Network shape: {}'.format(network_structure))
    print('Iris data classification score: {}%'.format(round(accuracy,2)))

    # load pima indian diabetes dataset
    filename = 'pima-indians-diabetes.csv'
    df = pd.read_csv(filename,header=None)

    # normalize data
    df_x = df.iloc[:,:-1].apply(pd.to_numeric).dropna() # convert features to numerical data
    # computes normalized numerical data
    compute_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
    # join with labels column
    df_norm = compute_norm.join(df.iloc[:,len(df.columns)-1])

    # replace labels column with 2 prediction outputs: [1,0] or [0,1]
    df_norm['labels'] = df_norm.iloc[:,-1:] # add labels column
    df_norm_pred = pd.concat([df_norm.iloc[:,:-2],df_norm.iloc[:,-1]],axis=1)

    # split into training and test sets
    df_shuffled = df_norm_pred.sample(frac=1) # shuffle dataset
    test_data = df_shuffled.iloc[:int((1/5)*df_shuffled.shape[0]),:]
    train_data = df_shuffled.iloc[int((1/5)*df_shuffled.shape[0]):,:]
    test_data_x = np.array(test_data.iloc[:,:-1])
    test_data_y = test_data.iloc[:,-1:]
    train_data_x = np.array(train_data.iloc[:,:-1])
    train_data_y = train_data.iloc[:,-1:]

    # process data for training
    train_target = train_data_y.values
    test_target = test_data_y.values

    # create the network class, structure is (8,4,1) (1 node in output layer)
    network_structure = (8,4,1)
    mlp = MlpNetwork(network_structure)

    itermax = 5000 # max iterations
    lmerr = 1e-6 # min error rate as stopping condition
    for i in range(itermax+1):
        err = mlp.train_network(train_data_x, train_target, momentum = 0.7)
        if err <= lmerr:
            print("Desired error reached. Iter: {0}".format(i))
            break

    # network output
    net_output = mlp.run_network(test_data_x)

    # convert network output to binary - data processing for diabetes dataset
    min_o = min(net_output)
    max_o = max(net_output)
    mid = 1/2*(max_o - min_o)
    new_out = []
    for i in net_output:
        new_out.append((i - min_o)/(max_o - min_o))

    new_out_binary = []
    for i in new_out:
        if i >= 0.5:
            new_out_binary.append([1])
        else:
            new_out_binary.append([0])

    new_out_binary = np.array(new_out_binary)

    # compute accuracy
    accuracy = mlp.accuracy_score(new_out_binary,test_target)
    print('\nNetwork shape: {}'.format(network_structure))
    print('Diabetes data classification score: {}%'.format(round(accuracy,2)))


# # Part 4 - Experiments and Analysis

# #### Experiment with different parameters
#
# Varying the number of hidden layers and the number of nodes in each hidden layer

# In[24]:

print(' \n Part 4:\n','-' * len('Part 4:'))
def paramvary(filename,network_structure):
        if filename == 'iris.data':
            # load iris data for classification
            data = []
            with open(filename) as infile:
                    for line in infile.read().splitlines():
                        tokens = line.split(",")
                        data.append(tokens)
            # converts data to pandas dataframe called df
            df = pd.DataFrame(data)

            # normalize data
            df_x = df.iloc[:,:-1].apply(pd.to_numeric).dropna() # convert features to numerical data
            # computes normalized numerical data
            compute_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
            # join with labels column
            df_norm = compute_norm.join(df.iloc[:,len(df.columns)-1])

            # replace iris labels column with 3 prediction outputs: [1,0,0], [0,1,0], [0,0,1]
            df_norm['labels'] = df_norm.iloc[:,-1:] # add labels column
            global df_norm_pred
            df_norm_pred = df_norm.labels.map({'Iris-setosa':[1,0,0], # replace with predictions
                                           'Iris-versicolor':[0,1,0],
                                           'Iris-virginica':[0,0,1]}).to_frame()
            df_norm_pred = pd.concat([df_norm.iloc[:,:-2], df_norm_pred], axis=1) # remove extra col

            # split into training and test sets
            df_shuffled = df_norm_pred.sample(frac=1) # shuffle dataset
            test_data = df_shuffled.iloc[:int((1/5)*df_shuffled.shape[0]),:]
            train_data = df_shuffled.iloc[int((1/5)*df_shuffled.shape[0]):,:]
            test_data_x = np.array(test_data.iloc[:,:-1])
            test_data_y = test_data.iloc[:,-1:]
            train_data_x = np.array(train_data.iloc[:,:-1])
            train_data_y = train_data.iloc[:,-1:]

            # process data for training - both train and test labels
            train_target = []
            for i in train_data_y.values:
                train_target.append(i[0])
            train_target = np.array(train_target)

            test_target = []
            for i in test_data_y.values:
                test_target.append(i[0])
            test_target = np.array(test_target)

            # create network with structure (4,3,3)
            mlp = MlpNetwork(network_structure)

            # when to stop model training
            itermax = 100 # max iterations
            lmerr = 1e-6 # min error rate as stopping condition
            global error_score_iris
            error_score_iris = [] # for use to make training graphs

            for i in range(itermax+1):
                err = mlp.train_network(train_data_x, train_target, momentum = 0.7)
                error_score_iris.append(err)

                # stopping condition
                if err <= lmerr:
                    print("Desired error reached. Iter: {0}".format(i))
                    break

            # network output
            net_output = mlp.run_network(test_data_x)
            new_out = mlp.convert_out(net_output)

            # convert outputs to desired format for iris dataset
            accuracy = mlp.accuracy_score(new_out,test_target)

            # compute and print accuracy/ classification score
            print('\nNetwork shape: {}'.format(network_structure))
            print('Iris data classification score: {}%'.format(round(accuracy,2)))

        elif filename == 'pima-indians-diabetes.csv':
            # load pima indian diabetes dataset
            df = pd.read_csv(filename,header=None)

            # normalize data
            df_x = df.iloc[:,:-1].apply(pd.to_numeric).dropna() # convert features to numerical data
            # computes normalized numerical data
            compute_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
            # join with labels column
            df_norm = compute_norm.join(df.iloc[:,len(df.columns)-1])

            # replace labels column with 2 prediction outputs: [1,0] or [0,1]
            df_norm['labels'] = df_norm.iloc[:,-1:] # add labels column
            df_norm_pred = pd.concat([df_norm.iloc[:,:-2],df_norm.iloc[:,-1]],axis=1)

            # split into training and test sets
            df_shuffled = df_norm_pred.sample(frac=1) # shuffle dataset
            test_data = df_shuffled.iloc[:int((1/5)*df_shuffled.shape[0]),:]
            train_data = df_shuffled.iloc[int((1/5)*df_shuffled.shape[0]):,:]
            test_data_x = np.array(test_data.iloc[:,:-1])
            test_data_y = test_data.iloc[:,-1:]
            train_data_x = np.array(train_data.iloc[:,:-1])
            train_data_y = train_data.iloc[:,-1:]

            # process data for training
            train_target = train_data_y.values
            test_target = test_data_y.values

            # create the network class, structure is (8,4,1) (1 node in output layer)
            mlp = MlpNetwork(network_structure)

            itermax = 100 # max iterations
            lmerr = 1e-6 # min error rate as stopping condition

            global error_score_diabetes
            error_score_diabetes = []
            for i in range(itermax+1):
                err = mlp.train_network(train_data_x, train_target, momentum = 0.7)
                error_score_diabetes.append(err)
                if err <= lmerr:
                    print("Desired error reached. Iter: {0}".format(i))
                    break

            # network output
            net_output = mlp.run_network(test_data_x)

            # convert network output to binary - data processing for diabetes dataset
            min_o = min(net_output)
            max_o = max(net_output)
            mid = 1/2*(max_o - min_o)
            new_out = []
            for i in net_output:
                new_out.append((i - min_o)/(max_o - min_o))

            new_out_binary = []
            for i in new_out:
                if i >= 0.5:
                    new_out_binary.append([1])
                else:
                    new_out_binary.append([0])

            new_out_binary = np.array(new_out_binary)

            # compute accuracy
            accuracy = mlp.accuracy_score(new_out_binary,test_target)
            print('\nNetwork shape: {}'.format(network_structure))
            print('Diabetes data classification score: {}%'.format(round(accuracy,2)),'\n')

# iris - get classification score for different network params (number of layers and nodes in each layer)
paramvary('iris.data',network_structure = (4,5,3))
paramvary('iris.data',network_structure = (4,5,6,3))
paramvary('iris.data',network_structure = (4,4,3))
paramvary('iris.data',network_structure = (4,8,2,3))

# diabetes dataset
paramvary('pima-indians-diabetes.csv',network_structure = (8,4,1))
paramvary('pima-indians-diabetes.csv',network_structure = (8,8,6,1))
paramvary('pima-indians-diabetes.csv',network_structure = (8,5,1))
paramvary('pima-indians-diabetes.csv',network_structure = (8,6,2,1))


# #### Compare with existing MLP implementations

# Compare with existing MLP implementations using sklearn.

# In[25]:


from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

'''Iris'''
# Load dataset
iris = datasets.load_iris()

# Create feature matrix
X_iris = iris.data

# Create target vector
y_iris = iris.target

# validation split
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris)

# scaling the data
scaler_iris = StandardScaler()
scaler_iris.fit(X_train_iris)
X_train_iris = scaler_iris.transform(X_train_iris)
X_test_iris = scaler_iris.transform(X_test_iris)

# classifier - 'logistic' equivalent to sigmoid according to documentation
mlp_iris = MLPClassifier(hidden_layer_sizes=(5),activation='logistic')

# fit data
mlp_iris.fit(X_train_iris,y_train_iris)

# predictions
predictions_iris = mlp_iris.predict(X_test_iris)

# Compute and print accuracy
accuracy_iris = round(accuracy_score(y_test_iris, predictions_iris) * 100,2)
print('Iris data classification score using sklearn: {}%'.format(accuracy_iris))

'''Diabetes'''
# Load dataset
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)

dataset.columns = [
    "NumTimesPrg", "PlGlcConc", "BloodP",
    "SkinThick", "TwoHourSerIns", "BMI",
    "DiPedFunc", "Age", "HasDiabetes"]

# validation split
train_set, test_set = train_test_split(dataset)

# Separate labels from the rest of the dataset
train_set_labels = train_set["HasDiabetes"].copy()
train_set = train_set.drop("HasDiabetes", axis=1)

test_set_labels = test_set["HasDiabetes"].copy()
test_set = test_set.drop("HasDiabetes", axis=1)

# scaling
scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)

# classifier - 'logistic' equivalent to sigmoid according to documentation
mlp_diabetes = MLPClassifier(hidden_layer_sizes=(5),activation='logistic')

# fit data
mlp_diabetes.fit(train_set,train_set_labels)

# predictions
predictions_diabetes = mlp_diabetes.predict(test_set)

# Compute and print accuracy
accuracy_diabetes = round(accuracy_score(test_set_labels, predictions_diabetes) * 100,2)
print('Diabetes data classification score using sklearn: {}%'.format(accuracy_diabetes))


# #### Graphs of training progress

# In[36]:


import matplotlib.pyplot as plt

paramvary('iris.data',network_structure = (4,5,3))
plt.plot(error_score_iris)
plt.title("Error Rate for Iris Dataset")
plt.xlabel("Iterations")
plt.ylabel("Training Error")

plt.show()


# In[37]:


paramvary('pima-indians-diabetes.csv',network_structure = (8,5,1))
plt.plot(error_score_diabetes)
plt.title("Error Rate for Diabetes Dataset")
plt.xlabel("Iterations")
plt.ylabel("Training Error")

plt.show()

# #### Final remarks
#
# How to handle categorical data: convert categorical data to numerical data. For example, the iris dataset had 3 categorical labels representing each flower type; these were transformed into the following: [1,0,0], [0,1,0] and [0,0,1]. As a result, the iris MLP classifier required 3 nodes in the output layer.
#
# How many layers in the network: the number of layers in a network is arbitrary and is best found through experimentation. There should be 1 input layer, 1 output layer, and any number of hidden layers.
#
# How many nodes should be in each hidden layer: there can be any number of nodes in the hidden layers.
#
# How many nodes should be in the output layer: there should be as many nodes in the output layer as there are classes in the classification problem, e.g. for iris there are 3 nodes in the output layer.
#
# When should learning be stopped: when the error is 0 or set to a very low number. In this implementation learning would stop if the error is smaller than 1e-6.
#
# Overall approach: 1 network class for the MLP including the following main methods for running the network, training the network and calculating the sigmoid activation function and its derivative. The weight updates for back-propagation are tracked via weight matrices which get updates during forward and backward passes for each run of the algorithm.
#
# Toughest part of the assignment: implementing all the weight updates for a single pass (backward or forward) using matrices. This was overcome by representing the backpropagation algorithm as a succession of vector multiplications.
