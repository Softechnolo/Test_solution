from docx import Document
from vector_database import VectorDatabase
from embedding_model import EmbeddingModel
from openai_api import OpenAI_API
from text_splitter import TextSplitter

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#  this Load the text from the given docx file
def load_text_from_docx(file_path):
    doc = Document(file_path)
    text = ' '.join([p.text for p in doc.paragraphs])
    return text

# Split the text into chunks
def split_text(text, splitter):
    return splitter.split(text)

# Adding chunks to the vector database
def add_chunks_to_database(chunks, database):
    for chunk in chunks:
        vector = EmbeddingModel().embed(chunk)
        database.addData(vector)

# Creating a prompt using the process discussed above
def create_prompt(query, database):
    matches = database.find_best_matches(query)
    context = ' '.join(matches)
    prompt = f"{context}\n{query}"
    return prompt

# Get the answer from gpt-3 api
def get_answer_from_gpt3(prompt):
    return OpenAI_API().get_answer(prompt)

# Getting all the things together
def user_query(query):
    text = load_text_from_docx("DataLaw.docx")
    chunks = split_text(text, the_splitter)
    database = VectorDatabase(index_name="mohd")  # Create the VectorDatabase instance here
    add_chunks_to_database(chunks, database)
    prompt = create_prompt(query, database)
    answer = get_answer_from_gpt3(prompt)
    return answer


the_splitter = TextSplitter(chunk_size=100) 

if __name__ == '__main__':
    query = input("Please enter your query here: ")
    answer = user_query(query)
    print(f"Answer: {answer}")
