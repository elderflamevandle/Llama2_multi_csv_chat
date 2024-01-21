# MultiCSV Chat App


## Introduction
------------
The MultiCSV Chat App is a Python application that allows you to chat with multiple CSV documents for `World Happiness Dataset from year 2015- 2019`. You can ask questions about the CSVs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a LLAMA2 language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded CSVs.

## How It Works
------------

The application follows these steps to provide responses to your questions:

1. CSVs Loading: The app reads multiple CSV documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks. Used HUggingFace Embeddings for generating vector representation of text chuks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model(LLAMA 2), which generates a response based on the relevant content of the CSVs.

## Dependencies and Installation
----------------------------
To install the MultiCSV Chat App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```Command Line -->
model_path = path to your llm model (LLAMA2 in this case)
```

## Usage
-----
To use the MultiCSV Chat App, follow these steps:

1. Ensure that you have installed the required dependencies.

2. Run the final_app.py file  , to run this code on the local terminal. It works smoother with local terminal.

3. If using on local teminal please provide the path for Data files given in the data folder.

4. Rest o fthe things are automated. Just wait for processes to complete and then ask you questions.

5. Run the `final_app.py` file using the Streamlit CLI. To run it with frondend. `There are some issues with web application and it is runnign slow`. Execute the following command:
   ```
   streamlit run web_app2.py
   ```

6. The application will launch in your default web browser, displaying the user interface.

7. Load multiple CSV documents (all at once) into the app by following the provided instructions.

8. Ask questions in natural language about the loaded PDFs using the chat interface.


