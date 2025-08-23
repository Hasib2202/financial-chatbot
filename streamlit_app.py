import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import FinancialChatbot

def initialize_chatbot():
    """Initialize the chatbot and cache it in Streamlit session state."""
    if 'chatbot' not in st.session_state:
        with st.spinner('Initializing Financial Policy Chatbot...'):
            st.session_state.chatbot = FinancialChatbot("financial_policy_document.txt")
            st.session_state.chatbot.initialize()
    return st.session_state.chatbot

def initialize_chat_history():
    """Initialize chat history in session state."""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "👋 Hello! I'm your Financial Policy Assistant. I can help you understand the Territory's financial policy document, including information about budget, debt, infrastructure, taxation, and financial management principles. What would you like to know?"
            }
        ]

def main():
    st.set_page_config(
        page_title="Financial Policy Chatbot",
        page_icon="🏛️",
        layout="wide"
    )
    
    # Header
    st.title("🏛️ Financial Policy Chatbot")
    st.markdown("*Ask questions about the Territory's Financial Policy Objectives and Strategies*")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 About This Chatbot")
        st.markdown("""
        This AI-powered chatbot can answer questions about:
        
        - **Budget & Financial Results**
        - **Debt Management**
        - **Infrastructure Investment**
        - **Taxation Policy**
        - **Superannuation Funding**
        - **Financial Management Principles**
        
        The chatbot uses vector search to find relevant information and remembers conversation context.
        """)
        
        st.markdown("---")
        
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "What is the government's budget situation?",
            "Tell me about debt management",
            "What are the infrastructure projects?",
            "How is taxation managed?",
            "What about superannuation funding?",
            "Explain the financial principles"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.user_question = question
        
        st.markdown("---")
        
        # Conversation summary
        if st.button("📋 Show Conversation Summary"):
            if 'chatbot' in st.session_state:
                summary = st.session_state.chatbot.get_conversation_summary()
                st.text_area("Conversation Summary", summary, height=150)
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "👋 Hello! I'm your Financial Policy Assistant. What would you like to know?"
                }
            ]
            st.rerun()
    
    # Initialize components
    chatbot = initialize_chatbot()
    initialize_chat_history()
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the financial policy..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching financial policy document..."):
                try:
                    response = chatbot.ask(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Handle example questions from sidebar
    if hasattr(st.session_state, 'user_question'):
        prompt = st.session_state.user_question
        delattr(st.session_state, 'user_question')
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Searching financial policy document..."):
            try:
                response = chatbot.ask(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
            except Exception as e:
                error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, ChromaDB, and Sentence Transformers*")

if __name__ == "__main__":
    main()
