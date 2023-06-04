import streamlit as st
import torch
from torch.optim import Adam
from model.DKT.RNNModel import RNNModel
from Data.dataloader import getDataLoader
import ast
from Evaluation import eval
import matplotlib.pyplot as plt

def main():
    st.title('Knowledge Tracing Model')

    # Description
    st.markdown("""
        ## Welcome to the Knowledge Tracing Model

        This web application allows you to use a pre-trained Knowledge Tracing Model to make predictions on test data.

        ### Model Parameters

        Use the sidebar to customize the model's architecture and hyperparameters.

        - **Batch Size:** Configure the batch size for training and prediction.
        - **Number of Epochs:** Set the number of training epochs.

        ### Load Data

        Click the 'Load Data' button to load the training and test data. The data will be loaded using the specified batch size.

        ### Run Model

        Once the data is loaded (Data loaded successfully!), click the 'Run Model' button to start the prediction on the test data. The model will make predictions and display the results.

        ### Probability of correctness

        After training the model, you can estimate the probability of correctness for a single question.

        ---
    """)

    # Parameters
    input_dim = 202
    hidden_dim = 100
    layer_dim = 2
    output_dim = 101
    batch_size = st.sidebar.number_input('Batch Size', min_value=1, step=1, value=64)
    num_epochs = st.sidebar.number_input('Number of Epochs', min_value=1, step=1, value=10)

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, device)
    model.load_state_dict(torch.load('Result/model.pth'))
    model.to(device)

    # Load the data
    train_loader, test_loader = None, None

    if st.sidebar.button('Load Data'):
        with st.spinner("Loading data..."):
            train_loader, test_loader = getDataLoader(batch_size, output_dim, hidden_dim)
        st.success("Data loaded successfully!")

    # Perform prediction on the test data and display the results
    if st.sidebar.button('Run Model') and train_loader is not None and test_loader is not None:
        st.subheader("Prediction")
        prediction = eval.test_epoch(model, test_loader, eval.lossFunc(output_dim, hidden_dim, device), device)
        st.markdown(f"<p style='font-size: 18px;'>{prediction}</p>", unsafe_allow_html=True)
    
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

    # Estimate the probability of correctness for a single question
    if st.sidebar.button('Probability of correctness'):
        st.subheader("Estimate Probability of Correctness")

        question_index = st.number_input("Enter the question index (from 1 to 100):", min_value=1, max_value=100, step=1)

        if question_index and test_loader is not None:
            probability = calculate_probability(model, question_index, test_loader)
            st.write(f"Probability of correctness for question {question_index}: {probability}")

    # Training loop
    if st.sidebar.button('Start Training') and train_loader is not None:
        st.subheader("Training Progress")
        optimizer = Adam(model.parameters())
        loss_func = eval.lossFunc(output_dim, hidden_dim, device)

        training_loss = []
        evaluation_auc = []
        evaluation_f1 = []
        evaluation_recall = []
        evaluation_precision = []

        st.write("Training progress:")
        for epoch in range(num_epochs):
            # Perform training
            model, optimizer, epoch_loss = eval.train_epoch(model, train_loader, optimizer, loss_func, device)
            training_loss.append(epoch_loss)

            # Perform evaluation
            auc, f1, recall, precision = eval.evaluate_model(model, test_loader, loss_func, device)
            evaluation_auc.append(auc)
            evaluation_f1.append(f1)
            evaluation_recall.append(recall)
            evaluation_precision.append(precision)

            # Display epoch information on Streamlit
            st.write(f"Epoch {epoch+1} completed")
            st.write(f"AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")

        # Visualize training loss and evaluation metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(training_loss)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 1].plot(evaluation_auc)
        axes[0, 1].set_title("Evaluation AUC")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("AUC")
        axes[1, 0].plot(evaluation_f1)
        axes[1, 0].set_title("Evaluation F1 Score")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 1].plot(evaluation_recall, label="Recall")
        axes[1, 1].plot(evaluation_precision, label="Precision")
        axes[1, 1].set_title("Evaluation Recall and Precision")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()

        st.subheader("Training Loss and Evaluation Metrics")
        st.pyplot(fig)

if __name__ == '__main__':
    main()
