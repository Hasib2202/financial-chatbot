# 🏛️ Financial Policy Chatbot - AI Assessment

**Join Venture AI Assessment Test**  
**Candidate:** Hasanmahmudnayeem3027@gmail.com  
**Repository:** https://github.com/Hasib2202/financial-chatbot.git

An AI-powered chatbot that answers questions about financial policy documents using vector search and conversation memory. Built with Python, ChromaDB, and Streamlit, this chatbot demonstrates advanced NLP capabilities including document processing, semantic search, and context-aware responses.

## ✨ Key Features

- **🔍 Smart Document Processing**: Automatic extraction and structuring of financial data
- **🧠 Vector Database Search**: Semantic similarity search using ChromaDB and Sentence Transformers
- **💬 Conversation Memory**: Context-aware responses that maintain conversation flow
- **🎨 Professional Web Interface**: Modern Streamlit app with streaming responses
- **📊 Response Formatting**: Topic-specific formatting for budget, debt, infrastructure, taxation, and risk assessment
- **📖 Interactive Demo**: Comprehensive Jupyter notebook demonstrating all features

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8+** (Tested with Python 3.13.7)
- **Git** (for cloning the repository)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Hasib2202/financial-chatbot.git
cd financial-chatbot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the web interface:**
```bash
streamlit run streamlit_app_professional.py
```

4. **Open your browser and go to:**
```
http://localhost:8501
```

## 📚 How to Use

### Web Interface
1. Start the Streamlit app using the command above
2. Ask questions about financial policy in the chat interface
3. Use the "📋 Generate Executive Summary" button for conversation analysis
4. Try questions like:
   - "What is the budget situation?"
   - "Tell me about debt management"
   - "What are the financial risks?"
   - "How does taxation relate to GSP?"

### Command Line
```python
from chatbot import FinancialChatbot

# Initialize chatbot
chatbot = FinancialChatbot("financial_policy_document.txt")

# Ask a question
response = chatbot.ask("What is the government's budget situation?")
print(response)
```

### Jupyter Notebook Demo
Open `financial_chatbot_demo.ipynb` in Jupyter Lab/Notebook to see:
- Complete implementation walkthrough
- Interactive examples
- Technical demonstrations
- Performance analysis

## 📁 Project Structure

```
financial-chatbot/
├── 📄 chatbot.py                      # Main chatbot implementation
├── 🌐 streamlit_app_professional.py   # Web interface
├── 📓 financial_chatbot_demo.ipynb    # Interactive demonstration
├── 📋 financial_policy_document.txt   # Source document
├── 📦 requirements.txt                # Dependencies
├── 📖 README.md                       # This file
├── 📘 USER_GUIDE.md                   # Detailed usage guide
└── 🗄️ chroma_db/                      # Vector database (auto-created)
```

## 🛠️ Technical Architecture

### Core Components

1. **DocumentProcessor**: Extracts and structures financial data from policy documents
2. **VectorDatabase**: ChromaDB-based semantic search with Sentence Transformers
3. **ConversationMemory**: Maintains context and enhances follow-up questions
4. **FinancialChatbot**: Main orchestrator providing contextual responses

### Key Technologies

- **🤖 AI/ML**: Sentence Transformers (all-MiniLM-L6-v2)
- **🗄️ Vector DB**: ChromaDB 1.0.20 (local, persistent)
- **🔤 Text Processing**: LangChain RecursiveCharacterTextSplitter
- **🌐 Web Framework**: Streamlit 1.48.1
- **🐍 Python**: 3.8+ compatible

## 💡 Example Interactions

### Budget Analysis
```
👤 User: "What is the budget situation?"
🤖 Bot: "## 💰 BUDGET ANALYSIS
        Executive Summary:
        • 2005-06 Budget Position: Strategic deficit of $91.5m
        • Recovery Strategy: Efficiency measures implemented
        • Policy Framework: Balanced budget over economic cycle..."
```

### Risk Assessment
```
👤 User: "What are the financial risks?"
🤖 Bot: "## ⚖️ FINANCIAL RISK ASSESSMENT
        Executive Summary:
        • Risk Management Framework: Comprehensive approach
        • Regulatory Compliance: Financial Management Act 1996
        • Strategic Objective: Prudent fiscal risk management..."
```

## 🎯 Assessment Requirements Met

✅ **Extract Data**: Financial information extracted and structured from policy document  
✅ **Vector Database**: ChromaDB with semantic search implemented  
✅ **Conversation Memory**: Context-aware responses with conversation tracking  
✅ **Clear Responses**: Professional formatting with source citations  
✅ **Web Interface**: Modern Streamlit application with excellent UX  

## Overview

This chatbot is designed to help users understand complex financial policy documents by providing:

- **Intelligent Document Search**: Uses vector embeddings to find relevant information
- **Conversation Memory**: Remembers context and previous questions
- **Source Citations**: Provides references to document sections
- **Multiple Interfaces**: Command-line and web-based options

## Features

### 📄 Document Processing

- Extracts and processes financial policy information
- Identifies key sections (budget, debt, infrastructure, taxation, etc.)
- Preserves source information for citations
- Extracts financial data (amounts, percentages, years)

### 🔍 Vector Search Database

- Uses ChromaDB for efficient similarity search
- Sentence Transformers for semantic embeddings
- Stores document chunks with metadata
- Fast retrieval of relevant information

### 🧠 Conversation Memory

- Tracks conversation history and topics
- Enhances vague queries with context
- Maintains coherent dialogue flow
- Provides conversation summaries

### 💬 Smart Response Generation

- Context-aware responses based on question type
- Formatted answers for different topics (budget, debt, etc.)
- Source citations for transparency
- Helpful tips and explanations

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd financial-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Chatbot

#### Option A: Command Line Interface

```bash
python chatbot.py
```

#### Option B: Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

The web interface will open in your browser at `http://localhost:8501`

## Usage Examples

### Example Questions You Can Ask:

**Budget & Financial Results:**

- "What is the government's budget situation?"
- "Is the budget in surplus or deficit?"
- "Tell me about the financial forecasts"

**Debt Management:**

- "How does the government manage debt?"
- "What are the debt levels?"
- "What about borrowings and interest?"

**Infrastructure:**

- "What infrastructure projects are planned?"
- "How is capital investment managed?"
- "Tell me about new construction projects"

**Taxation:**

- "How is taxation managed?"
- "What are the tax policies?"
- "What about GSP and tax burden?"

**Superannuation:**

- "What about superannuation funding?"
- "How are pension liabilities managed?"

**Financial Principles:**

- "What are the financial management principles?"
- "Explain the government's financial objectives"

### Sample Conversation:

```
💬 Your question: What is the budget situation?

🤖 Response:
Based on the financial policy document:

• The 2005-06 Budget is in deficit, but the Government has introduced measures to return to surplus.
• The Government aims to maintain a balanced budget over the economic cycle.
• The Budget provides an aggregate surplus over four years.

📄 Source: Section 2, Section 3

💡 The document emphasizes that short-term deficits are acceptable as long as there's a surplus over the complete economic cycle.
```

## Technical Architecture

### Core Components:

1. **DocumentProcessor**:

   - Loads and processes the financial policy document
   - Extracts sections and metadata
   - Identifies financial data and keywords

2. **VectorDatabase**:

   - ChromaDB for vector storage and search
   - Sentence Transformers for embeddings
   - Efficient similarity search

3. **ConversationMemory**:

   - Tracks conversation history
   - Maintains topic context
   - Enhances user queries

4. **FinancialChatbot**:
   - Main orchestrator class
   - Coordinates all components
   - Generates contextual responses

### Data Flow:

```
User Query → Memory Enhancement → Vector Search → Response Generation → Memory Update
```

## Project Structure

```
financial-chatbot/
├── chatbot.py              # Main chatbot implementation
├── streamlit_app.py        # Web interface
├── financial_policy_document.txt  # Source document
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── chroma_db/             # Vector database (created automatically)
```

## Dependencies

- **chromadb**: Vector database for similarity search
- **sentence-transformers**: Text embeddings
- **langchain**: Text processing utilities
- **streamlit**: Web interface
- **python-dotenv**: Environment management

## How the Chatbot "Remembers" Things

The conversation memory system works in several ways:

1. **Topic Tracking**: Identifies the main topic of each question (budget, debt, etc.)
2. **Context Enhancement**: Improves vague follow-up questions by adding context
3. **History Maintenance**: Keeps track of recent questions and responses
4. **Contextual Responses**: Uses conversation history to provide more relevant answers

**Example:**

```
User: "What about the budget?"
Bot: [Provides budget information]

User: "What about debt?"  # Bot knows this is about debt specifically
Bot: [Provides debt information]

User: "Tell me more"      # Bot enhances this with previous context
Enhanced Query: "Tell me more regarding debt"
Bot: [Provides additional debt details]
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_app_professional.py
```

### Production Deployment
The application is designed to run locally without external API dependencies:
- No OpenAI API required
- Local vector database (ChromaDB)
- Offline embeddings (Sentence Transformers)
- Self-contained Python environment

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Document Not Found**: Verify `financial_policy_document.txt` exists
   ```bash
   ls financial_policy_document.txt
   ```

3. **Streamlit Port Conflict**: Use alternative port
   ```bash
   streamlit run streamlit_app_professional.py --server.port 8502
   ```

## 🤝 Contributing

This project is designed for assessment purposes. For questions or clarifications:
- **Email**: hasanmahmudnayeem3027@gmail.com
- **GitHub**: https://github.com/Hasib2202/financial-chatbot

## 📄 License

This project is created for educational and assessment purposes as part of the Join Venture AI evaluation process.

---

## 🎯 Assessment Summary

This Financial Policy Chatbot demonstrates:

- **✅ Technical Proficiency**: Advanced Python, AI/ML, and web development
- **✅ Problem Solving**: Effective document processing and information retrieval
- **✅ User Experience**: Professional interface with excellent usability
- **✅ Code Quality**: Clean, well-documented, and maintainable architecture
- **✅ Innovation**: Context-aware conversation memory and professional response formatting

**Built with ❤️ for Join Venture AI Assessment**

---

**Repository**: https://github.com/Hasib2202/financial-chatbot.git  
**Submission**: hasanmahmudnayeem3027@gmail.com  
**Date**: August 23, 2025

## Search Strategy

The vector search is designed for optimal retrieval:

1. **Semantic Embeddings**: Uses sentence transformers to understand meaning
2. **Chunk Strategy**: Documents split into 1000-character chunks with 200-character overlap
3. **Metadata Enrichment**: Each chunk tagged with section type, keywords, financial data
4. **Multi-Result Search**: Retrieves top 3 most relevant sections
5. **Source Tracking**: Maintains reference to original document sections

## Customization

### Adding New Document Types:

1. Update `DocumentProcessor._identify_section_type()` for new section types
2. Add new response formatting methods
3. Update keyword lists in `ConversationMemory`

### Modifying Response Format:

- Edit the `_format_*_response()` methods in `FinancialChatbot`
- Customize the response templates for different question types

### Adjusting Search Parameters:

- Modify `chunk_size` and `chunk_overlap` in `DocumentProcessor`
- Adjust `n_results` in search queries
- Change embedding model in `VectorDatabase`

## Development Notes

### Key Design Decisions:

1. **ChromaDB Choice**: Local, persistent vector database that doesn't require external services
2. **Sentence Transformers**: Lightweight, efficient embeddings that work offline
3. **Chunk Strategy**: Balance between context preservation and search precision
4. **Memory Design**: Simple but effective conversation tracking
5. **Response Formatting**: Topic-specific templates for better user experience

### Future Enhancements:

- [ ] Support for multiple document types
- [ ] Advanced query understanding with NLP
- [ ] Export conversation summaries
- [ ] Integration with external APIs
- [ ] Multilingual support

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Memory Issues**: For large documents, consider reducing chunk size
3. **Slow Responses**: ChromaDB creates indexes on first run - subsequent queries are faster
4. **Missing Document**: Ensure `financial_policy_document.txt` is in the same directory

### Performance Tips:

- The first query takes longer as the system builds indexes
- Vector database persists between runs for faster startup
- Use the web interface for better user experience
- Clear conversation history if memory usage becomes high

## Contributing

Feel free to improve this chatbot by:

1. Adding support for more document formats
2. Improving response quality
3. Adding new features like export functionality
4. Enhancing the user interface

## License

This project is created for educational and assessment purposes.

---

**Built with ❤️ for Join Venture AI Assessment**

_Demonstrates AI development skills including document processing, vector search, conversation management, and user interface design._
