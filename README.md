# Deep Maximum Entropy Markov Model

This is an implementation of a Deep Maximum Entropy Markov Model in PyTorch, that uses the Viterbi algorithm for inference to solve the task of Named Entity Recognition.

You must have pytorch, nltk, numpy, and scipy installed. It uses python 2.7. 

You can run the program like so to load the pre trained parameters, and run inference using Viterbi.
`python deep_memm.py --best_parameters saved_model.pt`

In order to train the model, do: 
`python deep_memm.py`

This will save the best parameters in a file `model.pt` and perform inference as well. 

It does not use GPU, and takes 5 minutes on my Macbook Pro to run with the parameters loaded. 

The F-Score is printed, and I get ~65.0. The accuracy of the neural network used is 90%, and it is a simple Feed Forward Neural Network with two Fully Connected Layers. The reason for this is to optimize performance.
