
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../train')

import torch
from model import LSTMClassifier


# global variables used for testing
batch_size = 8
sequence_length = 100
embedding_dim = 16
hidden_dim = 150
vocab_size = 500
out_features = 1  #linear layer constructor parameter


tensors = None


def test_get_latest_tensor_shapes():

	lstm = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size)

	input_tensor = torch.randint(low=0, high=5, size=(batch_size, 1+sequence_length), dtype=torch.long)

	output = lstm(input_tensor)

	tensor_shapes = lstm.get_latest_tensor_shapes()

	global tensors

	tensors = tensor_shapes

	assert(len(tensor_shapes) == 12)

	print("Test passed for *test_get_latest_tensor_shapes*\n")


def test_embedding_layer():

	'''
	Input: (*), LongTensor of arbitrary shape
	Output: (*, H), where * is the input shape and H = embedding_dim

	'''
	embedding_input = tensors['Embedding_Input']
	embedding_output = tensors['Embedding_Output']

	assert(embedding_input == torch.Size([sequence_length, batch_size]))
	assert(embedding_output == torch.Size([sequence_length, batch_size, embedding_dim]))

	print("Test passed for *test_embedding_layer*\n")


def test_lstm_layer():

	'''
	Input: Tensor of shape (seq_len, batch, input_size) i.e. (100, 8, embedding_dim); this is why we x = x.t() and slice the tensor
    Output: Tensor of shape (seq_len, batch, num_directions * hidden_size) i.e. (100, 8, 1*150)

	'''

	lstm_input = tensors['LSTM_Input']
	lstm_output = tensors['LSTM_Output']

	assert(lstm_input == torch.Size([sequence_length, batch_size, embedding_dim]))
	assert(lstm_output == torch.Size([sequence_length, batch_size, hidden_dim]))

	print("Test passed for *test_lstm_layer*\n")


def test_dropout_layer():

	'''
	Input: No change in tensor shape; same as LSTM output
	Output: No change in tensor shape; same as LSTM output

	'''

	dropout_input = tensors['Dropout_Input']
	dropout_output = tensors['Dropout_Output']

	assert(dropout_input == torch.Size([sequence_length, batch_size, hidden_dim]))
	assert(dropout_output == torch.Size([sequence_length, batch_size, hidden_dim]))

	print("Test passed for *test_dropout_layer*\n")


def test_linear_layer():

	'''
	Input: Linear layer input shape is arbitrary; the last dimension of the input tensor is in_features layer parameter.
		   This layer reduces the last dimension of the input tensor (in_features) to out_features layer parameter.
	Output: (N, *, out_features) where all but the last dimension are the same shape as the input

	'''

	linear_input = tensors['Linear_Input']
	linear_output = tensors['Linear_Output']

	assert(linear_input == torch.Size([sequence_length, batch_size, hidden_dim]))
	assert(linear_output == torch.Size([sequence_length, batch_size, out_features]))

	print("Test passed for *test_linear_layer*\n")


def test_sigmoid_layer():

	'''
	Input: (N, *) where * means any number of additional dimensions
	Output: (N, *), same shape as the input

	'''

	sigmoid_input = tensors['Sigmoid_Input']
	sigmoid_output = tensors['Sigmoid_Output']

	assert(sigmoid_input == torch.Size([batch_size]))
	assert(sigmoid_output == torch.Size([batch_size]))

	print("Test passed for *test_sigmoid_layer*\n")


# Not currently needed; we are dissecting the forward pass layer by layer tensor flow
def test_forward_pass():

	'''
	batch_size: number of reviews passed into the forward method at once (from train_loader object); we test a single batch individually
	sequence_length: sequence of integers representing the first n words in the input review

	'''

	lstm = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size)

	# input are train_loader batches
	# test forward pass with one batch of size (8, 1+100)
	input_tensor = torch.randint(low=0, high=5, size=(batch_size, 1+sequence_length), dtype=torch.long) #torch.rand throws exception

	assert(input_tensor.shape == (batch_size, 1+sequence_length))

	# each row of the train batch has the form (batch_size, 1+review[n]) where 1 is the actual review length
	# and review[n] is a sequence of integers representing the first n words in the review.

	output = lstm(input_tensor)

	assert(output.shape == torch.Size([input_tensor.shape[0]]))



print("\nTesting...\n")

test_get_latest_tensor_shapes()
test_embedding_layer()
test_lstm_layer()
test_dropout_layer()
test_linear_layer()
test_sigmoid_layer()


# IT'S SHOCKING THAT THE FORWARD PASS WON'T ALWAYS THROW EXCEPTIONS FOR TENSOR SHAPE 'MISMATCHES'
# MODEL LAYERS DON'T KNOW THE MEANING OF THE DATA, JUST SHAPES AND TYPES
# WE MUST DEFINITELY TEST MODEL METHODS, CUSTOM TRAINING LOOPS, ETC. - EVERYTHING THOROUGHLY

