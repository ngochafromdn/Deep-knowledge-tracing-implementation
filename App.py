
import streamlit as st
import numpy as np
import time
import random
import json

# Load the predictions array here
sample = random.randint(0, 20)
batch = random.randint(0, 63)

with open(f'Result/prediction_{sample}.json', 'r') as file:
    predict_data_json = file.read()

# Convert the JSON data back to a Python list
predictions = json.loads(predict_data_json)
predictions = predictions['prediction'][batch] 
def main(prediction):
    st.set_page_config(page_title="Question Accuracy Probability", page_icon="📚")
    st.title("Question Accuracy Probability based on Deep Knowledge Tracing model")
    
    st.markdown(
        """
        Welcome to the Question Accuracy Probability App!
        
        This app predicts the accuracy probability for the next question based on the previous questions and their accuracy. The model behind it is based on the Deep Knowledge Tracing model with Assistmentdata2015.
        
        For more information, please go to : https://github.com/ngochafromdn/Deep-knowledge-tracing-implementation
        
        Suppose you are a students answer 100 questions from the questions list, this app will help you predict the accuracy probability of your next questions based on the result of previous questions 
        
    
        To use the app, follow these steps:
        
        1. Enter a list of previous question indices in the format [1, 2, 3]. The indices should be within the range of 1 to 100.
        2. Enter the previous question accuracy list in the format [0, 1, 0]. Use 0 to indicate an incorrect answer and 1 to indicate a correct answer. The list should have the same length as the previous question indices.
        3. Select the next question index from the dropdown menu. The index should be within the range of 1 to 100.
        4. Click the **Submit** button to get the accuracy probability for the next question.
        
        Try it out now!
        """
    )
    
    st.markdown("---")
    
    accuracy_prob = 1.0
    list_previous_question = st.text_input("Write a list of Previous Question Index (e.g., [1, 2, 3], Your answer should include the sign '[' and ']' ")
    previous_questions_accuracy_list = st.text_input("Previous Question Accuracy List, note that list list have the same length with previous questions list (0 means incorrect, 1 means correct), for example [1,0,1]. Your answer should include the sign '[' and ']'")
    next_question = st.selectbox("Next Question Index (1 - 100)", range(1, 101), index=0)
    submitted = st.button("Submit")
    
    if submitted:
        with st.spinner("Processing..."):
            # Simulating some processing time
            time.sleep(2)
            previous_questions_accuracy_list = eval(previous_questions_accuracy_list)
            list_previous_question = eval(list_previous_question)
            min_prob = 1.0
            max_prob = 0.0
            pre_sample = 0
            pre_batch = 0
            pre_step = 0
            
            for question_index in range(len(list_previous_question)):
                step = list_previous_question[question_index]-1
                accuracy_index = previous_questions_accuracy_list[question_index]
                if accuracy_index == 1:
                    accuracy_prob_this = max(prediction[step])
                else:
                    accuracy_prob_this = 1- max(prediction[step])
                
                accuracy_prob *= accuracy_prob_this
            
            accuracy_prob = predictions[list_previous_question[question_index]-1][next_question] * (accuracy_prob_this/5+1)
            
            if next_question in previous_questions_accuracy_list:
                if previous_questions_accuracy_list.index(next_question)==1:
                    if accuracy_prob < 0.5:
                        accuracy_prob = accuracy_prob*2
            if accuracy_prob > 1:
                accuracy_prob = accuracy_prob/2

            st.text(f"Accuracy Probability for the Next Question: {accuracy_prob}")

if __name__ == "__main__":
    main(predictions)

