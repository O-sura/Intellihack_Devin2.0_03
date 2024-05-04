import json
import os
import openai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv, dotenv_values 


load_dotenv() 

openai.api_key = os.getenv("OPENAI_APIKEY")

# Load JSON data
with open('./mnt/data/data.json') as f:
    data = json.load(f)

# Function to extract and flatten loan schemes with better structure
def extract_loan_schemes(data):
    chunks = []

    # Extract LoanTypes as a separate chunk
    loan_types = data["LoansDescription"]["LoanTypes"]
    loan_types_description = f"The loan types we have are {', '.join(loan_types)}."
    chunks.append(loan_types_description)
    
    # Extract loan schemes with a common template
    loan_schemes = data["LoansDescription"]["LoanSchemes"]
    
    for scheme in loan_schemes:
        scheme_type = scheme.get("LoanType", "")
        description = scheme.get("Description", "")
        features = " | ".join(scheme.get("Features", []))
        
        subtypes = scheme.get("SubTypes", [])
        if subtypes:
            for subtype in subtypes:
                subtype_name = subtype.get("SubLoanType", "")
                subtype_description = subtype.get("Description", "")
                subfeatures = " | ".join(subtype.get("Features", []))
                quantum_of_loan = " | ".join(subtype.get("QuantumOfLoan", {}).get("Description", []))
                eligibility = " | ".join(subtype.get("Eligibility", []))
                rates_of_advance = subtype.get("RatesOfAdvance", [])
                rates_texts = []
                for rate in rates_of_advance:
                    rate_purpose = rate.get("Purpose", "")
                    rate_text = f"{rate_purpose}: {rate.get('Rate', '')}"
                    sub_purposes = rate.get("SubPurposes", [])
                    for sub_purpose in sub_purposes:
                        sub_rate_text = f"{sub_purpose['SubPurpose']}: {sub_purpose['Rate']}"
                        rates_texts.append(f"{rate_purpose}, {sub_rate_text}")
                    rates_texts.append(rate_text)
                rates_summary = " | ".join(rates_texts)
                
                sub_description = (
                    f"Subtype: {subtype_name}, Description: {subtype_description}, Features: {subfeatures}, "
                    f"Quantum Of Loan: {quantum_of_loan}, Eligibility: {eligibility}, Rates Of Advance: {rates_summary}"
                )
                chunks.append(f"Loan Type: {scheme_type}, Description: {description}, Features: {features}, {sub_description}")
        else:
            chunks.append(f"Loan Type: {scheme_type}, Description: {description}, Features: {features}")

    return chunks

# Extract text chunks from the JSON data
chunks = extract_loan_schemes(data)

# Function to generate embeddings using OpenAI
def generate_openai_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        # Generate embedding using OpenAI API
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        embeddings.append({'chunk': chunk, 'embedding': embedding})
    return embeddings

# Generate embeddings for the extracted chunks
vectors = generate_openai_embeddings(chunks)

# Initialize Chroma client and collection
client = chromadb.Client()
collection = client.create_collection("loan_embeddings")

# Add embeddings to the collection
for i, vector in enumerate(vectors):
    collection.add(
        documents=[vector['chunk']],
        embeddings=[vector['embedding']],
        ids=[str(i)]
    )

# Function to query the database
def query_chroma_db(query_text, top_n=3):
    # Generate the embedding for the query text
    response = openai.Embedding.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    query_embedding = response["data"][0]["embedding"]

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )

    # Return results
    return results

# Function to refine results with OpenAI GPT-3.5
def refine_with_openai(prompt, results):
    documents = "\n\n".join(results['documents'][0])
    combined_prompt = f"Act as a virtual assistant in a bank. here is the user given prompt:{prompt}\n\nRelevant Documents:\n{documents}\n\nProvide the best answer based on the documents:,And when answering you should refer to the information available in the data and anything not related to loans or loan support, please discard them with an appropriate message. is the user types \"thank you for your support\", End the conversation with an appropriate greeting and all."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": combined_prompt}
        ],
        max_tokens=200
    )
    
    return response.choices[0].message['content'].strip()

# Terminal input for queries
print("Enter your questions to query the database. Type 'exit' to quit.")
while True:
    query_text = input("\nYour question: ")
    if query_text.lower() == 'exit':
        break
    
    results = query_chroma_db(query_text, top_n=3)
    
    refined_answer = refine_with_openai(query_text, results)
    
    print(f"Answer: {refined_answer}")
