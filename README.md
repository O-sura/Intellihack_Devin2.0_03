# Finance Chatbot

This repository contains two different implementations of a finance chatbot that assists users in getting information about loan products and eligibility. Both implementations leverage OpenAI's GPT models to deliver meaningful responses based on provided data.

## Implementation 1: `finance_chatbot.py`

### Overview

This implementation uses OpenAI's GPT-3.5-turbo model to simulate a professional bank officer. The chatbot assists users with various loan-related queries, including:

- Checking eligibility for different types of loans based on criteria like credit score, income, employment status, and existing debts.
- Providing information on loan products such as features, interest rates, repayment terms, and eligibility requirements.
- Guiding users through the loan application process, including documentation and approval timelines.
- Offering personalized loan product recommendations and eligibility tips based on the user's financial situation.

### Setup and Usage

1. **Install Dependencies**: Install the required libraries with:
    ```bash
    pip install openai gradio python-dotenv
    ```
2. **Configure Environment**: Create a `.env` file in the project directory and add your OpenAI API key:
    ```ini
    OPENAI_APIKEY=your-api-key
    ```
3. **Add Data**: Ensure the `data.json` file is present in the same directory.
4. **Run the Chatbot**: Start the chatbot with:
    ```bash
    python finance_chatbot.py
    ```
5. **Interact with the Bot**: The chatbot will launch a web-based interface where you can ask questions related to loans.

## Implementation 2: `finance_chatbot_enhanced.py`

### Overview

This implementation uses a more sophisticated approach by leveraging OpenAI's embeddings and ChromaDB to store and retrieve loan data efficiently. The chatbot combines embeddings with GPT-3.5-turbo to provide context-aware answers to user queries.

### Features and Scalability

- **Advanced Data Handling**: Extracts and processes detailed loan information from the provided `data.json`.
- **Efficient Search with Embeddings**: Uses OpenAI's `text-embedding-ada-002` model to generate embeddings for the data, enabling quick and context-aware search.
- **Scalable Storage and Retrieval**: Stores and retrieves embeddings using ChromaDB, which is designed to scale with larger datasets for fast and accurate search.
- **Enhanced Query Refinement**: Refines answers using OpenAI's GPT-3.5-turbo model based on the context retrieved from the database.

### Setup and Usage

1. **Install Dependencies**: Install the required libraries with:
    ```bash
    pip install openai chromadb python-dotenv
    ```
2. **Configure Environment**: Create a `.env` file in the project directory and add your OpenAI API key:
    ```ini
    OPENAI_APIKEY=your-api-key
    ```
3. **Add Data**: Ensure the `data.json` file is present in the same directory.
4. **Run the Chatbot**: Start the chatbot with:
    ```bash
    python finance_chatbot_enhanced.py
    ```
5. **Interact via Terminal**: The chatbot will prompt for user input in the terminal. Type your queries related to loans and get personalized responses based on the data.

## Conclusion

Both implementations aim to provide useful financial insights to users regarding loans and banking products. The first implementation is simpler and ideal for quick setups, while the enhanced version offers improved scalability and context-aware responses through advanced embedding-based search using ChromaDB.
