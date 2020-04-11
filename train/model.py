
import torch.nn as nn

from collections import OrderedDict


class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, dropout=0.5):
        """
        Initialize the model by setting up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)

        self.sig = nn.Sigmoid()
        
        self.word_dict = None

        self.latest_tensor_shapes = OrderedDict()


    def forward(self, x):
        """
        Perform a forward pass of our model on some input tensor.
        """

        #print(f"Shape of input tensor: {x.shape}\n")
        self.latest_tensor_shapes['Model_Input'] = x.shape

        x = x.t()
        #print(f"Shape of transposed tensor: {x.shape}\n")

        lengths = x[0,:]
        #print(f"Shape of lengths var: {lengths.shape}\n")

        reviews = x[1:,:]
        #print(f"Shape of reviews var: {reviews.shape}\n")

        ##------------------------------------------------------###

        self.latest_tensor_shapes['Embedding_Input'] = reviews.shape

        embeds = self.embedding(reviews) # requires input tensor of dtype torch.long
        #print(f"Shape of Embedding layer output: {embeds.shape}\n")

        self.latest_tensor_shapes['Embedding_Output'] = embeds.shape

        ##------------------------------------------------------###

        self.latest_tensor_shapes['LSTM_Input'] = embeds.shape

        lstm_out, _ = self.lstm(embeds)
        
        #print(f"Shape of LSTM layer output: {lstm_out.shape}\n")
        self.latest_tensor_shapes['LSTM_Output'] = lstm_out.shape

        ##------------------------------------------------------###

        self.latest_tensor_shapes['Dropout_Input'] = lstm_out.shape

        out = self.dropout(lstm_out)
        #print(f"Shape of Dropout layer output: {out.shape}\n")

        self.latest_tensor_shapes['Dropout_Output'] = out.shape

        ##------------------------------------------------------###

        self.latest_tensor_shapes['Linear_Input'] = out.shape

        out = self.dense(out)
        #print(f"Shape of Linear layer output: {out.shape}\n")

        self.latest_tensor_shapes['Linear_Output'] = out.shape

        ##------------------------------------------------------###

        out = out[lengths - 1, range(len(lengths))]
        #print(f"Shape of transformed linear output: {out.shape}\n")

        self.latest_tensor_shapes['Linear_Output_Transformed'] = out.shape

        out = out.squeeze()
        #print(f"Shape of transformed linear output: {out.shape}\n")

        ##------------------------------------------------------###

        self.latest_tensor_shapes['Sigmoid_Input'] = out.shape

        final = self.sig(out)
        #print(f"Shape of final Sigmoid output: {final.shape}\n")

        self.latest_tensor_shapes['Sigmoid_Output'] = final.shape

        return final


    def initialize_hidden_state(self):

        # TODO: implement weight initialization for recurrent state

        pass


    def get_latest_tensor_shapes(self):

        return self.latest_tensor_shapes
