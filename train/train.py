import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'], model_info['dropout'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size: int, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def _get_val_data_loader(batch_size: int, val_dir):
    print("Get validation data loader.")

    val_data = pd.read_csv(os.path.join(val_dir, "validation.csv"), header=None, names=None)

    val_y = torch.from_numpy(val_data[[0]].values).float().squeeze()
    val_X = torch.from_numpy(val_data.drop([0], axis=1).values).long()

    val_ds = torch.utils.data.TensorDataset(val_X, val_y)

    return torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

def validate(model, val_loader):
    
    ''' Function to compute validation metrics during model training. 
    
    Args: 
    
    model: Model object
    val_loader: Validation data loader object
    
    Returns: total validation loss and the model object
    
    '''
    
    # Get validation loss
    total_val_loss = 0
                
    # Switch model to evaluation mode (no dropout, batchnorm, etc)
    model.eval()
                
    # Turn off gradients
    with torch.no_grad():
                
        for val_batch in val_loader:

            val_batch_X, val_batch_y = val_batch
            
            val_batch_X = val_batch_X.to(device)
            val_batch_y = val_batch_y.to(device)

            val_output = model(val_batch_X)
            
            val_loss = loss_fn(val_output, val_batch_y)
            
            total_val_loss += val_loss.data.item()
            
    return total_val_loss, model


def train(model, train_loader, val_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    counter = 0
    print_every = 100
    
    for epoch in range(1, epochs + 1):
        
        model.train()
        
        total_loss = 0
        
        for batch in train_loader:
            
            counter += 1
            
            # PyTorch accumulates gradients. We need to clear them out before each batch pass.
            model.zero_grad()
            
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            
            # Run the forward pass for a given batch
            output = model(batch_X)
            
            # Compute the loss for a given batch
            loss = loss_fn(output, batch_y)
            
            # Accumulate the loss for each batch to get the total loss per epoch
            total_loss += loss.data.item()
        
            # Compute the gradients
            loss.backward()
            
            # Update the model's trainable parameters
            optimizer.step()
            
            # Validation check
            if counter % print_every == 0:
                
                # Compute validation metrics
                total_val_loss, model = validate(model, val_loader)

                # Switch model back to training mode
                model.train()
                
                print("Epoch: {}/{}...".format(epoch, epochs),
                      "Step: {}...".format(counter),
                      "BCELoss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(total_val_loss / len(val_loader)))

        # For each epoch, log the average loss per batch
        print("Epoch: {}, BCELoss: {:.6f}, Val Loss: {:.6f}".format(epoch, total_loss / len(train_loader), total_val_loss / len(val_loader)))


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='dropout layer probability (default: 0.5)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    #parser.add_argument('--val-data-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    
    # Load the validation data.
    val_loader = _get_val_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size, args.dropout).to(device)

    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}, dropout {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size, args.dropout
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, val_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'dropout': args.dropout,
        }
        torch.save(model_info, f)

	# Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
