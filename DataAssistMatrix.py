import csv
from collections import defaultdict

class DataAssistMatrix:
    def __init__(self, params):
        print('Loading')
        trainPath =  'train-data.csv'
        self.trainData = self.load_data(trainPath)
        testPath = 'test-data.csv'
        self.testData = self.load_data(testPath)
        self.questions = self.get_questions()
        self.n_questions = len(self.questions)
    
    def load_data(self, file_path):
        data =[]
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Get header row
            n_columns = len(header)
            n_rows = sum(1 for _ in reader)  # Count the number of data rows
            file.seek(0)  # Reset the file pointer to the beginning
            for _ in range((n_rows + 1) // 3):  # Include the header row in the range
                student = {}
                for _ in range(3):
                    row = next(reader)
                    if _ == 1:
                        question_id = row
                    elif _ == 0:
                        correct = row[0]
                    elif _ == 2:
                        n_answers = row
                        if int(correct) >= 2:
                            student['questionId'] = student.get('questionId', []) + [question_id]
                            student['n_answers'] = student.get('n_answers', []) + [n_answers]
                            student['correct'] = correct
                if 'questionId' in student:
                        data.append(student)
        return data



    
    def get_questions(self):
        questions = set(range(1, 101))  # Create a set of numbers from 1 to 100
        return questions

    
    def get_train_data(self):
        return self.trainData
    
    def get_test_data(self):
        return self.testData
    
    def get_test_batch(self):
        return self.testData.copy()

# Example usage
params = {}
data_matrix = DataAssistMatrix(params)
train_data = data_matrix.get_train_data()
test_data = data_matrix.get_test_data()
test_batch = data_matrix.get_test_batch()

