# Student Name: Oğuzhan Özer
# Student ID: 260201039

import numpy as np
import pandas as pd

import util

# HYPERPARAMETERS
input_size = 7602
output_size = 10
hidden_layers_sizes = 100
learning_rate = 0.01
number_of_epochs = 5

train_data_path = "./data/drugLibTrain_raw.tsv"
test_data_path = "./data/drugLibTest_raw.tsv"

# initial weights and biases
W_B = {
    'W1': np.random.randn(hidden_layers_sizes, input_size) * np.sqrt(2/hidden_layers_sizes),
    'b1': np.ones((hidden_layers_sizes, 1)) * 0.01,
    'W2': np.random.randn(hidden_layers_sizes, hidden_layers_sizes) * np.sqrt(2/hidden_layers_sizes),
    'b2': np.ones((hidden_layers_sizes, 1)) * 0.01,
    'W3': np.random.randn(output_size, hidden_layers_sizes) * np.sqrt(2/hidden_layers_sizes),
    'b3': np.ones((output_size, 1)) * 0.01
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activation_function(layer):
    # the activation function for hidden layers
    return sigmoid(layer)


def derivation_of_activation_function(signal):
    return signal * (1 - signal)


def loss_function(true_labels, probabilities):
    # sum-of-squares error loss
    difference = true_labels - probabilities['Y_pred']
    return rss(difference)


def rss(layer):
    # sum-of-squares error (rss) is used to turn activations into probability distribution
    return np.sum(layer ** 2)


def derivation_of_loss_function(true_labels, probabilities):
    # the derivation should be with respect to the output neurons
    return 2 * (probabilities["Y_pred"] - true_labels)


def forward_pass(data):
    # Calculate the Z for the hidden layer 1
    z1 = np.dot(data, W_B['W1'].T) + W_B['b1'].T
    # Calculate the activation output for the hidden layer 1
    a1 = activation_function(z1)
    # Calculate the Z for the hidden layer 2
    z2 = np.dot(a1, W_B['W2'].T) + W_B['b2'].T
    # Calculate the activation output for the hidden layer 2
    a2 = activation_function(z2)
    # Calculate the Z for the output layer
    z3 = np.dot(a2, W_B['W3'].T) + W_B['b3'].T
    # Calculate the activation output for the output layer (linear activation)
    y_pred = z3
    # Save hidden layer output and z's in a dictionary
    forward_results = {"Z1": z1,
                       "A1": a1,
                       "Z2": z2,
                       "A2": a2,
                       "Y_pred": y_pred}
    return forward_results


# [hidden_layers] is not an argument, so replace it with your desired hidden layers
def backward_pass(input_layer, output_layer, loss):
    # calculate layer deltas
    output_delta = loss
    z3_delta = np.dot(output_delta, W_B['W3'])

    a2_delta = z3_delta * derivation_of_activation_function(output_layer['A2'])
    z2_delta = np.dot(a2_delta, W_B['W2'])

    a1_delta = z2_delta * derivation_of_activation_function(output_layer['A1'])
    z1_delta = np.dot(a1_delta, W_B['W1'])

    # update weights and biases
    W_B['W3'] -= learning_rate * np.outer(output_layer['A2'], output_delta).T
    W_B['b3'] -= learning_rate * np.sum(output_delta, axis=1, keepdims=True)
    
    W_B['W2'] -= learning_rate * np.outer(output_layer['A1'], a2_delta).T
    W_B['b2'] -= learning_rate * np.sum(a2_delta, axis=1, keepdims=True)

    W_B['W1'] -= learning_rate * np.outer(input_layer, a1_delta).T
    W_B['b1'] -= learning_rate * np.sum(a1_delta, axis=1)


def train(train_data, train_labels, valid_data, valid_labels):
    # array initializations for accuracy and loss plot
    accuracy_list = np.array([])
    loss_list = np.array([])

    for epoch in range(number_of_epochs):
        index = 0

        # Same thing about [hidden_layers] mentioned above is valid here also
        for data, labels in zip(train_data, train_labels):
            predictions = forward_pass(data)
            loss_signals = derivation_of_loss_function(labels, predictions)
            backward_pass(data, predictions, loss_signals)

            if index % 400 == 0:  # at each 400th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_data, valid_labels)
                print(f"Epoch= {epoch}, Coverage= %{100*(index/len(train_data))}, Accuracy= {accuracy}, Loss= {loss}")
                # print("Epoch= "+str(epoch)+", Coverage= %" + str(100*(index/len(train_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))

            index += 1

    return accuracy_list, loss_list


def test(test_data, test_labels):

    avg_loss = 0
    predictions = []
    labels = []

    for data, label in zip(test_data, test_labels):  # Turns through all data
        fpr = forward_pass(data)
        predictions.append(fpr["Y_pred"])
        labels.append(label)
        avg_loss += np.sum(loss_function(label, fpr))

    # Maximum likelihood is used to determine which label is predicted, highest prob. is the prediction
    # And turn predictions into one-hot encoded

    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for i in range(len(predictions)):
        one_hot_predictions[i][np.argmax(predictions[i])] = 1

    predictions = one_hot_predictions

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score,  avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
        # if 1 is in same index with ground truth
        if np.argmax(predictions[i]) == np.argmax(true_labels[i]):
            true_pred += 1

    return true_pred / len(predictions)


if __name__ == "__main__":

    train_data = pd.read_csv(train_data_path, sep='\t')
    test_data = pd.read_csv(test_data_path, sep='\t')
    # use train_data['commentsReview'] or concatenate benefitsReview, sideEffectsReview, and commentsReview
    train_x = train_data['commentsReview']
    train_y = train_data['rating']
    # use test_data['commentsReview'] or concatenate benefitsReview, sideEffectsReview, and commentsReview
    test_x = test_data['commentsReview']
    test_y = test_data['rating']

    # creating one-hot vector notation of labels. (Labels are given numeric in the dataset)
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))

    for i in range(len(train_y)):
        new_train_y[i][train_y[i]-1] = 1

    for i in range(len(test_y)):
        new_test_y[i][test_y[i]-1] = 1

    train_y = new_train_y
    test_y = new_test_y

    # Training and validation split. (%75-%25)
    valid_x = np.asarray(train_x[int(0.75*len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.75*len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.75*len(train_x))])
    train_y = np.asarray(train_y[0:int(0.75*len(train_y))])

    # find word set in the training set and count words in each sample for inputs
    unique_words = util.find_unique_words(train_x)
    train_x = util.construct_input_values(train_x, unique_words)

    # count words in validation samples
    valid_x = util.construct_input_values(valid_x, unique_words)

    train(train_x, train_y, valid_x, valid_y)
    print("Test Scores:")

    # count words in test samples
    test_x = util.construct_input_values(test_x, unique_words)
    print(test(test_x, test_y))
