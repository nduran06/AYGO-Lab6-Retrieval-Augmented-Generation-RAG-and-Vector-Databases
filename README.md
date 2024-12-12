# AYGO-Lab6-Retrieval-Augmented-Generation-RAG-and-Vector-Databases

## Summary

* Using LangChain, OpenAI, and Pinecone

This lab demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using LangChain, OpenAI, and Pinecone. The system is designed to enhance language model responses by incorporating relevant information from a knowledge base.

## Overview

The RAG system consists of three main components:

1. **Document Processing Pipeline:**
   - Loads documents from text files
   - Splits documents into manageable chunks using RecursiveCharacterTextSplitter
   - Processes text while maintaining semantic coherence

2. **Vector Storage and Retrieval:**
   - Uses OpenAI embeddings to convert text chunks into vector representations
   - Implements two storage solutions:
     - FAISS for local vector storage
     - Pinecone for cloud-based vector storage
   - Enables efficient similarity search for relevant context retrieval

3. **Response Generation:**
   - Utilizes ChatOpenAI (GPT-4) for generating contextual responses
   - Implements a custom prompt template for consistent output
   - Combines retrieved context with user queries for accurate answers

## Installation Instructions

1. **Environment Setup**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows

# Install required packages
pip install langchain langchain_openai openai pinecone-client python-dotenv faiss-cpu
```

2. **API Keys Configuration**
Create a `.env` file in your project root with the following:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

## Code Structure and Implementation

The implementation is divided into several key functions:

### 1. Environment Setup
```python
def env_setup():
    """Set up required environment variables"""
    load_dotenv()
    requiredVars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missingVars = [var for var in requiredVars if not os.getenv(var)]
    if missingVars:
        raise ValueError(f"Missing required env variables: {', '.join(missingVars)}")
```

### 2. Document Processing
```python
def load_and_process_documents(file_path):
    """Load and split documents into chunks"""
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

### 3. Vector Store Implementation

The project implements two vector store solutions:

#### FAISS (Local Storage)
```python
def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store
```

#### Pinecone (Cloud Storage)
```python
def create_vector_store_pinecone(chunks, index_name="aygo-langchain-index2"):
    """Create a Pinecone vector store from document chunks"""
    embeddings = OpenAIEmbeddings()
    index = get_create_pinecone_index(index_name)
    vector_store = Pinecone.from_documents(
        index_name=index_name,
        embedding=embeddings,
        documents=chunks
    )
    return vector_store
```

### 4. RAG Chain Setup
```python
def setup_rag_chain(vector_store):
    """Create a retrieval chain using the vector store"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer based on the context, just say you don't know.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain
```

## Key Features and Improvements

1. **Robust Error Handling:**
   - Validates environment variables before execution
   - Implements proper error handling for API calls
   - Provides meaningful error messages for troubleshooting

2. **Flexible Vector Storage:**
   - Supports both local (FAISS) and cloud-based (Pinecone) vector storage
   - Enables easy switching between storage solutions
   - Maintains consistent interface for both implementations

3. **Optimized Text Processing:**
   - Uses recursive character splitting for better chunk coherence
   - Implements chunk overlap to maintain context across splits
   - Configurable chunk sizes for different use cases

4. **Enhanced Response Generation:**
   - Custom prompt template for consistent output format
   - Context-aware response generation
   - Fallback handling for unknown queries

## Usage Example

```python
# Initialize the RAG system
env_setup()
create_sample_data()
chunks = load_and_process_documents("ai_knowledge.txt")

# Choose vector store implementation
vector_store = create_vector_store_pinecone(chunks)  # or create_vector_store(chunks)

# Setup and use the RAG chain
chain = setup_rag_chain(vector_store)
response = query_rag(chain, "What is Artificial Intelligence?")
print(response['result'])
```

## Performance Comparison

### FAISS vs. Pinecone

1. **FAISS Advantages:**
   - No external API calls required
   - Lower latency for small datasets
   - Simpler setup and deployment

2. **Pinecone Advantages:**
   - Scales better with large datasets
   - Persistent storage across sessions
   - Distributed architecture for better reliability
   - Real-time updates and consistency


