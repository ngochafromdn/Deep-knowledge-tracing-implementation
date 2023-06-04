
"""
Usage:
    run.py rnn --hidden=<h> [options]

Options:
    --length=<int>                      max length of attemps [default: 100]
    --questions=<int>                   num of question [default: 101]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 42]
    --epochs=<int>                      number of epochs [default: 10]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimension of hidden state [default: 128]
    --layers=<int>                      layers of rnn [default: 2]
    --dropout=<float>                   dropout rate [default: 0.1]
"""


import os
import random
import logging
import torch

import torch.optim as optim
import numpy as np

from datetime import datetime
from docopt import docopt
from Data.dataloader import getDataLoader
from Evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = docopt(__doc__)
    length = int(args['--length'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])
    dropout = float(args['--dropout'])
    model_type = 'RNN'

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainLoader, testLoader = getDataLoader(bs, questions, length)

    from model.DKT.RNNModel import RNNModel
    model = RNNModel(questions * 2, hidden, layers, questions, device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = eval.lossFunc(questions, length, device)

    predicted_var = None  # Initialize a variable to store the predicted variable

    
    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer,
                                          loss_func, device)
        logger.info(f'epoch {epoch}')
        eval.test_epoch(model, testLoader, loss_func, device)

    # Save the model
    model_dir = 'Result'  
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    



if __name__ == '__main__':
    main()


   
