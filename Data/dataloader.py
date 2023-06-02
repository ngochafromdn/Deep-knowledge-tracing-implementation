import torch
import torch.utils.data as Data
from Data.readdata import DataReader


def getDataLoader(batch_size, num_of_questions, max_step):
    handle = DataReader('Raw/data.csv',
                        'Raw/test-data.csv', max_step,
                        num_of_questions)
    dtrain = torch.tensor(handle.getTrainData().astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(handle.getTestData().astype(float).tolist(),
                         dtype=torch.float32)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader

