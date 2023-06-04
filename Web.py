import streamlit as st
import torch
from torch.optim import Adam
from model.DKT.RNNModel import RNNModel
from Data.dataloader import getDataLoader
import ast
from Evaluation import eval

def main():
    st.title('Knowledge Tracing Model')

    # Description
    st.markdown("""
        ## Welcome to the Knowledge Tracing Model

        This web application allows you to estimate the probability of correctness for a single question using a pre-trained Knowledge Tracing Model.

        ### Model Parameters

        Use the sidebar to customize the model's architecture and hyperparameters.

        ### Load Data

        Click the 'Load Data' button to load the test data. The data will be used for estimating the probability of correctness.

        ### Probability of correctness

        After loading the data, you can estimate the probability of correctness for a single question.

        ### Trained Model Information

        To see information about the trained model, click the 'Model Information' button. It will display the architecture and summary of the model.

        ---
    """)

    # Parameters
    input_dim = 202
    hidden_dim = 100
    layer_dim = 2
    output_dim = 101

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model.load_state_dict(torch.load('Result/model.pth'))
    model.to(device)

    # Load the data
    test_loader = None
    data_loaded = False

    if st.sidebar.button('Load Data'):
        with st.spinner("Loading data..."):
            _, test_loader = getDataLoader(1, output_dim, hidden_dim)  # Set batch size to 1 for single question probability
            data_loaded = True
        st.success("Data loaded successfully!")

    # Estimate the probability of correctness for a single question
    if st.sidebar.button('Probability of correctness') and data_loaded:
        st.subheader("Estimate Probability of Correctness")

        question_index = st.number_input("Enter the question index (from 1 to 100):", min_value=1, max_value=100, step=1)

        if question_index:
            probability = calculate_probability(model, question_index, test_loader)
            st.write(f"Probability of correctness for question {question_index}: {probability}")

    # Trained Model Visualization
    if st.sidebar.button('Model Information'):
        st.subheader("Trained Model Information")
        st.text("Model Architecture:")
        st.code(model)

def calculate_probability(model, question_index, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the previous response for the specified question index from the test dataset
    for data in test_loader:
        questions = data['question']
        responses = data['response']
        break

    # Find the response for the specified question index
    prev_response = responses[question_index - 1]

    # Convert previous response to a numeric value
    response_value = int(prev_response)

    # Create an input tensor with the given response value
    input_tensor = torch.tensor([[response_value]]).to(device)

    # Pass the input tensor through the model to get the output probabilities
    with torch.no_grad():
        output_probs = model(input_tensor)

    # Extract the probability for the specified question index
    question_probability = output_probs[0][question_index].item()

    return question_probability

if __name__ == '__main__':
    main()
