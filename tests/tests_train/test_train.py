
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../train')

from train import model_fn
from train import _get_train_data_loader
from train import _get_val_data_loader
from train import validate
from train import train


def test_model_fn():

	# Assert 

	pass


def test_get_train_data_loader():

	# Assert 

	pass


def test_get_val_data_loader():

	# Assert 

	pass


def test_validate():

	# Assert 

	pass


def test_train():

	# Assert 

	pass


# A unit test is a test that covers a small unit of code, usually a function

'''

We want to test our functions in a way that is repeatable and automated. 
Ideally, we'd run a test program that runs all our unit tests and cleanly lets us know which ones failed and which ones succeeded. 
There are great tools available in Python that we can use to create effective unit tests: pytest	

The advantage of unit tests is that they are isolated from the rest of your program, and thus, no dependencies are involved. 
They don't require access to databases, APIs, or other external sources of information. 
However, passing unit tests isn’t always enough to prove that our program is working successfully. 

To show that all the parts of our program work with each other properly, 
communicating and transferring data between them correctly, we use integration tests.

'''

print("Train")
