# import numpy as np

# class Network(object):

#   def __init__(self, sizes):
#       self.num_layers = len(sizes)
#       self.sizes = sizes
#       self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#       self.weights = [np.random.randn(y, x)
#                           for x, y in zip(sizes[:-1], sizes[1:]) ]

#       # np.random.randn() generates Gaussian distribution with mean 0 and variance 1.

#       # zip is used to pair two lists.

#       """ Since the very first layer in our Network is input layer therefore we've omitted the first value from sizes 
#       when setting biases """

#       # This random initialisation will give our program a place to start from


# net = Network([2, 3, 1])

# print(net.weights[1])

# Third-party libraries

import random
import numpy as np
import mnist_loader
import mnist_custom_loader
import sys
import Tkinter

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                # write("Running Epoch %i" % (j))
                # write("Successfully classified %i / %i" % (self.evaluate(test_data), n_test))                
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                # write("Epoch %i complete" % (j))
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



global training_data
global validation_data
global test_data

def write(string):
    global text_box
    text_box.config(state=Tkinter.NORMAL)
    text_box.insert("end", string + "\n")
    text_box.see("end")
    text_box.config(state=Tkinter.DISABLED)

def load_dataset():
    global training_data
    global validation_data
    global test_data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    write("Dataset successfully loaded\n")    

def test_MNIST_data():
    global training_data
    global validation_data
    global test_data
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def load_custom_dataset():
    global training_data
    global validation_data
    global test_data
    # training_data, validation_data, sample_test_data = mnist_loader.load_data_wrapper()
    training_data, test_data = mnist_custom_loader.load_data_wrapper()
    write("Custom dataset successfully loaded\n")    

def test_custom_data():
    global training_data
    global test_data
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 2.0, test_data=test_data)

def close_window(): 
    root.destroy()

def center_window(width=300, height=200):
    # get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


root = Tkinter.Tk()
root.title('Optical Character Recognizer')

center_window(800, 600)

text_box = Tkinter.Text(root, state=Tkinter.DISABLED)
text_box.grid(row=0, column=1, columnspan=8)

button_1 = Tkinter.Button(root, text="Load Dataset", command=load_dataset)
button_1.grid(row=1, column=1)

button_2 = Tkinter.Button(root, text="Test MNIST dataset", command=test_MNIST_data)
button_2.grid(row=1, column=2)

button_3 = Tkinter.Button(root, text="Load Custom Dataset", command=load_custom_dataset)
button_3.grid(row=1, column=3)

button_4 = Tkinter.Button(root, text="Test Custom Dataset", command=test_custom_data)
button_4.grid(row=1, column=4)

button_5 = Tkinter.Button(root, text="Exit", command=close_window)
button_5.grid(row=1, column=5)


# net = Network([784, 30, 10])
# net.SGD(training_data, 1, 10, 3.0, test_data=test_data)


root.mainloop()
# print("Successfully trained. End of application")

