import streamlit as st
import torch
from torch.optim import Adam
from model.DKT.RNNModel import RNNModel
from Data.dataloader import getDataLoader
from Evaluation import eval

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

        Once the data is loaded, click the 'Run Model' button to start the prediction on the test data. The model will make predictions and display the results.

        ---
    """)

    # Parameters
    input_dim = 300
    hidden_dim = 50
    layer_dim = 4
    output_dim = 150
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
        prediction = eval.test_epoch(model, test_loader, eval.lossFunc(output_dim, hidden_dim, device), device)
        st.markdown("<h3>Prediction results:</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 18px;'>{prediction}</p>", unsafe_allow_html=True)

    # Training loop
    if train_loader is not None:
        optimizer = Adam(model.parameters())  # Initialize optimizer
        loss_func = eval.lossFunc(output_dim, hidden_dim, device)  # Define the loss function
        st.write("Training progress:")
        for epoch in range(num_epochs):
            # Perform training
            model, optimizer = eval.train_epoch(model, train_loader, optimizer, loss_func, device)

            # Display epoch information on Streamlit
            st.write(f"Epoch {epoch+1} completed")

            # Perform prediction on the test data and display the results after each epoch
            prediction = eval.test_epoch(model, test_loader, eval.lossFunc(output_dim, hidden_dim, device), device)


if __name__ == '__main__':
    main()
