from __future__ import division
import subprocess
import argparse
import nltk
import scipy
import numpy as np
import random
import sys
import gzip
import cPickle
#import _pickle as cPickle
# import torch
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time



class Classifier(object):
    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError("Train method not implemented")

    def inference(self):
        raise NotImplementedError("Inference method not implemented")

class MEMM(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, hidden_layer_size, output_class_size):
        super(MEMM, self).__init__()
        # set up the model
        self.word_embedding_size = word_embedding_size
        self.tag_embedding_size = word_embedding_size
        self.hidden_layer_size = hidden_layer_size
        self.output_class_size = output_class_size
        self.vocab_size = vocab_size

        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.tag_embedding_layer = nn.Embedding(self.output_class_size, self.word_embedding_size)
        self.fc1 = nn.Linear(self.word_embedding_size + self.word_embedding_size, self.hidden_layer_size)
        self.relu_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_class_size)
        self.relu_activation2 = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_layer_size // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_layer_size // 2)))

    def forward_pass(self, current_word, previous_tag):
        word_embeddings = self.word_embedding_layer(current_word)
        tag_embeddings = self.tag_embedding_layer(previous_tag)
        fc_input = torch.cat((word_embeddings, tag_embeddings), 1)
        out1 = self.fc1(fc_input)
        dropout_output = self.dropout(out1)
        activation1 = self.relu_activation(dropout_output)
        out2 = self.fc2(activation1)
        return out2

    def train(self, data_lex, data_y, optimizer, start_tag, train=False):
        total_count = 0
        accuracy = 0.0
        sum_correct_result = 0
        total_loss_list = []

        for i in np.arange(len(data_lex)):
            batch_sentences = data_lex[i]
            tag_vectors = data_y[i]

            total_count += len(batch_sentences)

            # create a batch from the sentence
            input_sentence = autograd.Variable(torch.from_numpy(batch_sentences).long())
            # the previous tags
            previous_tag_vector = np.asarray([start_tag] + list(tag_vectors[:-1]))
            input_targets = autograd.Variable(torch.from_numpy(previous_tag_vector).long())
            # the current tags
            output_targets = autograd.Variable(torch.from_numpy(tag_vectors).long())
            # training output
            output = self.forward_pass(input_sentence, input_targets)
            # calculate how many are correct
            max_index = output.max(dim = 1)[1]
            sum_correct_result = sum_correct_result + ((max_index == output_targets).sum().data.numpy()[0])

            if train == True:
                #loss
                l = F.cross_entropy(input=output, target=output_targets)

                total_loss_list.append(l.data.mean())
                # train
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

        if train == True:
            current_average_loss = np.mean(total_loss_list)
        accuracy = 100 * (sum_correct_result / total_count)

        if train == True:
            return current_average_loss, accuracy
        else:
            return accuracy

    def forward_inference(self, current_word, previous_tag):
        input_word = autograd.Variable(torch.from_numpy(np.asarray([current_word])).long())
        input_targets = autograd.Variable(torch.from_numpy(np.asarray([previous_tag])).long())
        output = self.forward_pass(input_word, input_targets)
        softmax_probs = F.softmax(output)
        return softmax_probs

    # Viterbi algorithm
    def inference(self, data_lex, num_words, num_states, start_tag):
        num_states = num_states - 1

        # table to store the probability of the most likely path so far
        table_1 = np.zeros((num_words, num_states))
        # table to store the backpointers of the most likely path so far
        table_2 = np.zeros((num_words, num_states))

        # initialization

        # get the first word from the data_lex (list of sentences) and make a batch that has all the states, all the initial states, and all the first words in 3 separate vectors to pass in for batch inference from the MLP
        probabilities = self.forward_inference(data_lex[0], start_tag).data.numpy()

        max_tag = np.argmax(probabilities, 1)[0]

        # for all states, table_1(1, s) = p(s | s_o, x1..x_m)
        # fill in the table using the start state given random initial and the current word
        for i in range(num_states):
            table_1[0, i] = probabilities[0, i]
            table_2[0, i] = 0


        for j in range(1, num_words):
            # create a transition matrix that stores the probability of each state given previous state and current word
            transition_matrix = []
            for l in range(num_states):
                output = self.forward_inference(data_lex[j], l)
                transition_matrix.append(output.data.numpy()[0])
            transition_matrix = np.asarray(transition_matrix)

            # i = current states
            for i in range(num_states):
                max_prob = 0
                max_index = 0
                # k = old states
                for k in range(num_states):
                    current_prob = table_1[j-1, k] * transition_matrix[k, i]
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_index = k

                        if max_index == 127:
                            print(max_prob)
                            print(k)
                            print(i)
                    max_prob = max(table_1[j-1, k] * transition_matrix[k, i], max_prob)
                # max over probability of a state in previous row * probability of going to this current state given that previous state
                table_1[j, i] = max_prob
                table_2[j, i] = max_index

        # for the last word, what is the state with the best probability
        maxPreviousTag = np.argmax(table_1[num_words - 1, :])

        reverse_final_tag_sequence = []
        reverse_final_tag_sequence.append(maxPreviousTag)
        for i in reversed(range(1, num_words)):
            maxPreviousTag = int(table_2[i, int(maxPreviousTag)])
            reverse_final_tag_sequence.append(maxPreviousTag)

        final_tag_sequence = []
        for i in reversed(reverse_final_tag_sequence):
            final_tag_sequence.append(i)


        return final_tag_sequence
                


def conlleval(p, g, w, filename='tempfile.txt'):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])

    return (precision, recall, f1score)

def preprocess_data():
    pass

# find the maximum value in a list of lists
def maxValue(inputlist):
    return max([sublist[-1] for sublist in inputlist])

def main():
    start_time = time.time()


    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")
    argparser.add_argument("--best_parameters")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    #train_set, valid_set, test_set, dicts = cPickle.load(f, encoding='latin1')

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    #print(valid_y)

    #idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    #idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())
    idx2label = dict((k,v) for v,k in dicts['labels2idx'].items())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].items())

    '''
    To have a look what the original data look like
    '''
    #print test_lex[0], map(lambda t: idx2word[t], test_lex[0])
    # print(test_lex[0]) 
    # print(list(map(lambda t: idx2word[t], test_lex[0])))
    # # #print test_y[0], map(lambda t: idx2label[t], test_y[0])
    #print(test_y[1])
    #print(list(map(lambda t: idx2label[t], test_y[1])))

    '''
    implement you training loop here
    '''
    model_save_path = "model.pt"
    vocab_size = int(maxValue(train_lex) + 2) 
    # + 1 to include the initial tag
    output_class_size = int(maxValue(train_y) + 1)
    start_tag = output_class_size
    word_embedding_size = 100
    hidden_layer_size = 256
    learning_rate = 0.001
    num_epochs = 5
    mymemm = MEMM(vocab_size, word_embedding_size, hidden_layer_size, output_class_size + 1)
    optimizer = optim.Adam(mymemm.parameters(), lr=learning_rate)


    # best parameters
    best_accuracy = 0.0
    best_test_accuracy = 0.0
    best_loss = 1000000
    # train the model to do tag classification when taking a previous tag and current word and predict what the tag should be

    if parsed_args.best_parameters is None:

        for epoch in range(num_epochs):
            train_loss, train_accuracy = mymemm.train(train_lex, train_y, optimizer, start_tag, train=True)
            validation_accuracy = mymemm.train(valid_lex, valid_y, optimizer, start_tag, train=False)
            test_accuracy = mymemm.train(test_lex, test_y, optimizer, start_tag, train=False)

            if train_accuracy >= best_test_accuracy:
                torch.save(mymemm, model_save_path)

            best_accuracy = max(best_accuracy, train_accuracy)
            best_loss = min(best_loss, train_loss)
            best_test_accuracy = max(best_test_accuracy, test_accuracy)

    else:
        # load the parameters
        mymemm = torch.load(parsed_args.best_parameters)
        # run for the test data
        test_accuracy = mymemm.train(test_lex, test_y, optimizer, start_tag, train=False)

    # run the viterbi algorithm for inference -> no training required
    #self, data_lex, num_words, num_states, start_tag
    mymemm.inference(test_lex[0], len(test_lex[0]), start_tag+1, start_tag)


    # '''
    # how to get f1 score using my functions, you can use it in the validation and training as well
    # '''
    predictions_test = [ map(lambda t: idx2label[t], mymemm.inference(x, len(x), start_tag+1, start_tag)) for x in test_lex ]
    groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y ]
    words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
    test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)

    print test_precision, test_recall, test_f1score

    elapsed_time = time.time() - start_time

    print(elapsed_time)



if __name__ == '__main__':
    main()
