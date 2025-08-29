# 🔧 API Documentation & Advanced Usage

## FinancialChatbot API Reference

### Core Classes

#### `FinancialChatbot(document_path: str)`

Main chatbot interface for financial policy queries.

**Parameters:**

- `document_path` (str): Path to financial policy document

**Methods:**

```python
ask(question: str) -> str
    """Ask a question and get formatted response"""

get_conversation_summary() -> str
    """Get summary of current conversation"""
```

#### `DocumentProcessor(document_path: str)`

Handles document processing and extraction.

**Methods:**

```python
process_document() -> List[Dict[str, any]]
    """Process document into structured chunks"""

extract_financial_data(text: str) -> Dict[str, any]
    """Extract financial data from text"""
```

#### `VectorDatabase(collection_name: str)`

Manages ChromaDB vector operations.

**Methods:**

```python
add_documents(documents: List[Dict]) -> None
    """Add documents to vector database"""

search(query: str, n_results: int = 3) -> List[Dict]
    """Semantic search for relevant content"""
```

#### `ConversationMemory(max_history: int = 10)`

Manages conversation context and memory.

**Methods:**

```python
add_interaction(question: str, response: str) -> None
    """Add Q&A pair to memory"""

enhance_question(question: str) -> str
    """Enhance vague questions with context"""

clear_history() -> None
    """Clear conversation history"""
```

## 🚀 Advanced Usage Examples

### Example 1: Custom Financial Document Processing

```python
from chatbot import DocumentProcessor, VectorDatabase, FinancialChatbot

# Process custom financial document
processor = DocumentProcessor("my_financial_report.txt")
documents = processor.process_document()

# Create custom vector database
vector_db = VectorDatabase("my_financial_db")
vector_db.add_documents(documents)

# Initialize chatbot with custom setup
chatbot = FinancialChatbot("my_financial_report.txt")
response = chatbot.ask("What are our quarterly results?")
```

### Example 2: Batch Query Processing

```python
# Process multiple queries
queries = [
    "What is the budget situation?",
    "Tell me about debt management",
    "What are the financial risks?",
    "How is infrastructure funded?"
]

results = []
for query in queries:
    response = chatbot.ask(query)
    results.append({
        'question': query,
        'answer': response,
        'topic': chatbot.memory.current_topic
    })
```

### Example 3: Custom Response Formatting

```python
# Access raw search results
search_results = chatbot.vector_db.search("budget deficit", n_results=5)

# Custom processing
for result in search_results:
    print(f"Content: {result['content'][:100]}...")
    print(f"Relevance: {result['score']:.3f}")
    print(f"Section: {result['metadata']['section_type']}")
```

### Example 4: Conversation Analysis

```python
# Analyze conversation patterns
summary = chatbot.get_conversation_summary()
context = chatbot.memory.get_context()

print(f"Current topic: {context['current_topic']}")
print(f"Questions asked: {context['history_length']}")
print(f"Recent topics: {context['recent_topics']}")
```

## 🔌 Integration Examples

### Streamlit Integration

```python
import streamlit as st
from chatbot import FinancialChatbot

@st.cache_resource
def load_chatbot():
    return FinancialChatbot("financial_policy_document.txt")

chatbot = load_chatbot()
user_input = st.chat_input("Ask about financial policy...")

if user_input:
    response = chatbot.ask(user_input)
    st.write(response)
```

### Flask API Integration

```python
from flask import Flask, request, jsonify
from chatbot import FinancialChatbot

app = Flask(__name__)
chatbot = FinancialChatbot("financial_policy_document.txt")

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    response = chatbot.ask(question)

    return jsonify({
        'question': question,
        'response': response,
        'topic': chatbot.memory.current_topic
    })
```

## ⚙️ Configuration Options

### Environment Variables

```bash
# Optional: Custom embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional: Database path
CHROMA_DB_PATH=./custom_chroma_db

# Optional: Chunk parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Advanced Configuration

```python
# Custom text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

custom_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " "]
)

# Custom embedding model
from sentence_transformers import SentenceTransformer
custom_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
```

## 🛠️ Debugging & Troubleshooting

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Debug vector search
results = chatbot.vector_db.search("budget", n_results=3)
logger.debug(f"Search results: {len(results)} documents found")
```

### Performance Monitoring

```python
import time

start_time = time.time()
response = chatbot.ask("What is the financial status?")
end_time = time.time()

print(f"Response time: {end_time - start_time:.3f} seconds")
```

---

**API Version:** 1.0  
**Last Updated:** August 30, 2025  
**Compatibility:** Python 3.8+
