import docx
import torch
from transformers import AutoModel, AutoTokenizer
from pinecone import Index
import openai
import nltk
import os

# Import PunktSentenceTokenizer
from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download NLTK resources for sentence segmentation (one-time execution)
nltk.download('punkt')

# Connect to Pinecone database (replace with your credentials)
index = Index(index_name="mohammed", environment="c1xkftx", host="https://mohammed-c1xkftx.svc.aped-4627-b74a.pinecone.io", api_key="ff9ec1d1-de35-45a4-bab7-cba9f33dbe4e")


def call_gpt3_api(prompt):
    """Calls the GPT-3 API and returns the generated text"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()


def load_and_split_text(docx_file):
    """Loads and splits text from a docx file"""
    document = docx.Document(docx_file)
    text = "\n".join([p.text for p in document.paragraphs])
    chunks = tokenizer.tokenize(text)
    return chunks


def embed_text(text):
    """Embeds text using the provided model"""
    tokenizer = AutoTokenizer.from_pretrained("openai/text-embedding-ada-002")
    model = AutoModel.from_pretrained("openai/text-embedding-ada-002")
    encoded_input = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded_input)
    # Ensure output is a list of numbers
    return output.last_hidden_state[:, 0, :].tolist()


def add_data_to_vector_db(data):
    embeddings = [embed_text(chunk) for chunk in data]
    # Create a dictionary where the keys are unique IDs and the values are the embeddings
    data_dict = {str(i): embedding.tolist() for i, embedding in enumerate(embeddings)}
    index.upsert(vectors=data_dict)



def get_similar_data(query):
    """Embeds the query and retrieves similar data from Pinecone"""
    query_embedding = embed_text(query)
    results = index.query(query_embedding)
    top_results = results["matches"]
    return [data[i] for i in top_results["ids"]]


def user_query(question):
    """Gets similar data, prompts GPT-3, and returns the answer and retrieved data"""
    relevant_data = get_similar_data(question)
    prompt = f"Answer the following legal question based on the provided context:\nQuestion: {question}\nContext:\n{'-'*20}\n{''.join(relevant_data)}\n{'-'*20}"
    answer = call_gpt3_api(prompt)
    return answer, relevant_data


docx_file = "DataLaw.docx"
data_chunks = load_and_split_text(docx_file)
add_data_to_vector_db(data_chunks)

while True:
    user_question = input("Ask your legal question (or type 'quit' to exit): ")
    if user_question.lower() == "quit":
        break
    answer, retrieved_data = user_query(user_question)
    print(f"Answer from GPT-3: {answer}")
    print(f"Retrieved Data Chunks:")
    for chunk in retrieved_data:
        print(chunk)

print("Exiting...")
