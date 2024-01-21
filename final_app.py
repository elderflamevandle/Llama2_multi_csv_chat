import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import sys

DB_FAISS_PATH = "vectorstore/db_faiss"

# Define directory and years
data_dir = "data/"
years = ["2015", "2016", "2017", "2018", "2019"]

# Create empty list to store combined data
combined_data = []

# Loop through each year
for year in years:
    # Read the year's CSV file
    file_path = os.path.join(data_dir, f"{year}.csv")
    df = pd.read_csv(file_path)

    # Add "year" column
    df["year"] = year

    # Append data to combined list
    combined_data.append(df)

# Combine all dataframes into one
combined_df = pd.concat(combined_data, ignore_index=True)

# Choose the output destination
output_path = "data/combined_data.csv"

# Save the combined data to a new CSV file
combined_df.to_csv(output_path, index=False)

print(f"Combined data saved to: {output_path}")

# Loading a CSV file
loader = CSVLoader(file_path = 'data/combined_data.csv', encoding = "utf-8", csv_args = {'delimiter': ','})
data = loader.load()

print(data)

# Splitting Data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
text_chunks = text_splitter.split_documents(data)
print("Text Chunks Completed")

# Download Sentence Transformer embeddings from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)
print(" --- Embeddings are stored into a vector database --- ")

# Import my language model
llm = CTransformers(
                    model="C:/Projects/Project Langchain/models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    config={
                        "max_new_tokens": 270,
                        "temperature": 0.1,
                    }
                    )

conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_turns=5)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=docsearch.as_retriever(),
    memory = conversational_memory
)

# Start chat loop
chat_history = []
p = 1 
while True:
    print("\n\n")
    print("Prompt No :", p)
    query = input(f"Input Prompt: ")

    if query == "exit":
        print("Exiting")
        sys.exit()

    if query == "":
        continue

    chat_history.append(query)
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append(result["answer"])

    print("Response:", result["answer"])
    p += 1