import argparse
from train import train
from test import test

# Check if run takes place in the main program
if __name__ == "__main__":
    # Create argument parser object
    args = argparse.ArgumentParser()
    # Adding arguments for the parser and setting default values
    args.add_argument('-lr', type=float, default=0.0001)
    args.add_argument('-gamma', type=float, default=0.7)
    args.add_argument('-lam', type=float, default=0.95)
    args.add_argument('-eps', type=float, default=0.1)
    args.add_argument('-c1', type=float, default=1.0)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-clip', type=float, default=0.1)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-batch_size', type=int, default=129)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-cicles', type=int, default=10000)
    args.add_argument('-train', default='False', choices=('True','False'))
    # Parse arguments
    arguments = args.parse_args()
    # Train agent if train is set to true
    if(arguments.train == 'True'):
        print('Status: In training')
        train(arguments)
    # Test agent if train is set to false
    else:
        print('Status: Testing')
        test(arguments)