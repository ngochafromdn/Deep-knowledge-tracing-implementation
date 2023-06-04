
import numpy as np
import itertools


class DataReader():
    def __init__(self, train_path, test_path, maxstep, numofques):
        self.train_path = train_path  # Path 
        self.test_path = test_path  # Path 
        self.maxstep = maxstep  # Maximum number of steps
        self.numofques = numofques  # Number of questions

    def getData(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            # Iterate over every three lines in the file (length of attemps, question, correctness)
            for len, ques, ans in itertools.zip_longest(*[file] * 3):
                len = int(len.strip().strip(','))  # Get the attemps' number value and convert it to an integer
                ques = [int(q) for q in ques.strip().strip(',').split(',')]  # Get the questions index array and convert each element to an integer
                ans = [int(a) for a in ans.strip().strip(',').split(',')]  # Get the correctness array and convert each element to an integer

                slices = len // self.maxstep + (1 if len % self.maxstep > 0 else 0)  # Calculate the number of slices needed based on the length and maxstep

                for i in range(slices):
                    temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])  # Create a temporary array of zeros

                    if len > 0:
                        if len >= self.maxstep:
                            steps = self.maxstep
                        else:
                            steps = len

                        # Fill the temporary array with ones based on the question and answer values
                        for j in range(steps):
                            if ans[i * self.maxstep + j] == 1:
                                temp[j][ques[i * self.maxstep + j]] = 1
                            else:
                                temp[j][ques[i * self.maxstep + j] + self.numofques] = 1

                        len = len - self.maxstep  # Update the remaining length

                    data.append(temp.tolist())  # Append the temporary array to the data list

            print('done: ' + str(np.array(data).shape))  # Print the shape of the processed data

        return data

    def getTrainData(self):
        print('loading train data...')
        trainData = self.getData(self.train_path)  # Load and process the training data
        return np.array(trainData)

    def getTestData(self):
        print('loading test data...')
        testData = self.getData(self.test_path)  # Load and process the test data
        return np.array(testData)
