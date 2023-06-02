
import streamlit as st
import pandas as pd
import torch
import torch.utils.data as Data
from Data.dataloader import getDataLoader
from model.DKT.RNNModel import RNNModel
from Evaluation.eval import performance, lossFunc

# Function to load and preprocess the data
def load_data(train_file, test_file, max_step, num_of_questions):
    handle = DataReader(train_file, test_file, max_step, num_of_questions)
    dtest = torch.tensor(handle.getTestData().astype(float).tolist(), dtype=torch.float32)
    test_loader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return test_loader

# Function to perform prediction
def predict(model, test_loader, num_of_questions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    prediction = torch.tensor([], device=device)
    ground_truth = torch.tensor([], device=device)

    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        prediction = torch.cat([prediction, pred])
        ground_truth = torch.cat([ground_truth, batch[:, :, :num_of_questions].sum(dim=1)])

    return prediction, ground_truth

# Function to display performance metrics
def display_performance(ground_truth, prediction):
    st.subheader('Performance Metrics')
    performance(ground_truth, prediction)

# Function to recommend items
def recommend_items():
    # Add your recommendation logic here
    # You can display recommended items based on the prediction or any other recommendation algorithm

    st.subheader('Recommendation')
    # Display recommended items


# Main Streamlit web application code
def main():
    st.title('DKT Model Evaluation and Recommendation System')
    st.sidebar.title('Configuration')

    # Read input files
    train_file = st.sidebar.file_uploader('Upload train data file (train-data.csv)', type='csv')
    test_file = st.sidebar.file_uploader('Upload test data file (test-data.csv)', type='csv')

    # Read parameters from user input or use default values
    max_step = st.sidebar.number_input('Max length of question sequence', value=50)
    num_of_questions = st.sidebar.number_input('Number of questions', value=150)
    input_dim = st.sidebar.number_input('Input dimension', value=300)
    hidden_dim = st.sidebar.number_input('Hidden dimension', value=50)
    layer_dim = st.sidebar.number_input('Number of layers', value=4)
    output_dim = st.sidebar.number_input('Output dimension', value=150)

    if train_file is not None and test_file is not None:
        # Load data
        test_loader = load_data(train_file, test_file, max_step, num_of_questions)

        # Perform prediction
        model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)
        model.load_state_dict(torch.load(model_path))
        prediction, ground_truth = predict(model, test_loader, num_of_questions)

        # Display performance metrics
        display_performance(ground_truth, prediction)

        # Recommend items
        recommend_items()

if __name__ == '__main__':
    main()

