import streamlit as st
import torch
from model.DKT.RNNModel import RNNModel
from Data.dataloader import getDataLoader

def main():
    st.title('Knowledge Tracing Model')

    # Description
    st.markdown("""
        ## Welcome to the Knowledge Tracing Model

        This web application allows you to estimate the probability of correctness for a single question using a pre-trained Knowledge Tracing Model.

        ### Probability of correctness

        Click the 'Probability of correctness' button to estimate the probability of correctness for a single question. Please wait for the data loader to load the necessary data.

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
    model_state_dict = torch.load('Result/model.pth')
    model.load_state_dict(model_state_dict)
    model.to(device)

    # Estimate the probability of correctness for a single question
    if st.button('Probability of correctness'):
        st.subheader("Estimate Probability of Correctness")

        question_index = st.number_input("Enter the question index (from 1 to 100):", min_value=1, max_value=100, step=1)

        if question_index:
            with st.spinner("Loading data..."):
                test_loader = getDataLoader(1, output_dim, hidden_dim)  # Set batch size to 1 for single question probability
                probability = calculate_probability(model, question_index, test_loader)
            st.write(f"Probability of correctness for question {question_index}: {probability}")

    # Trained Model Visualization
    if st.button('Model Information'):
        st.subheader("Trained Model Information")
        st.text("Model Architecture:")
        st.code(model)

def calculate_probability(model, question_index, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the previous response for the specified question index from the test dataset
    for data in test_loader:
        questions = data[:, :, :output_dim]
        responses = data[:, :, output_dim:]
        break

    # Find the response for the specified question index
    prev_response = responses[:, question_index - 1]

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

