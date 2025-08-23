import streamlit as st
import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import FinancialChatbot

def initialize_chatbot():
    """Initialize the chatbot with caching"""
    if 'chatbot' not in st.session_state:
        with st.spinner('Loading Financial Policy Analysis System...'):
            st.session_state.chatbot = FinancialChatbot("financial_policy_document.txt")
    return st.session_state.chatbot

def stream_response(response_text):
    """Stream response text for smooth display"""
    response_container = st.empty()
    
    # Split by sentences for more natural streaming
    sentences = response_text.split('. ')
    displayed_text = ""
    
    for sentence in sentences:
        if sentence.strip():
            # Add the sentence
            if displayed_text:
                displayed_text += ". " + sentence
            else:
                displayed_text = sentence
            
            # Add period if it's not the last sentence and doesn't end with punctuation
            if sentence != sentences[-1] and not sentence.endswith(('.', '!', '?', ':')):
                displayed_text += "."
            
            response_container.markdown(displayed_text)
            time.sleep(0.3)  # Pause between sentences
    
    # Final display with complete text
    response_container.markdown(response_text)
    return response_text

def main():
    # Page configuration
    st.set_page_config(
        page_title="Territory Financial Policy Analysis System",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        color: #333333 !important;
    }
    .metric-card h4 {
        color: #2a5298 !important;
        margin-bottom: 1rem;
    }
    .metric-card p, .metric-card div, .metric-card span {
        color: #333333 !important;
    }
    
    /* Executive Summary specific styling */
    .metric-card ul, .metric-card li {
        color: #333333 !important;
    }
    .stButton > button {
        background-color: #2a5298;
        color: white !important;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1e3c72;
        color: white !important;
    }
    
    /* Enhanced chat styling with better alignment */
    .stChatMessage {
        max-width: 80%;
        margin: 1rem 0;
    }
    
    /* User messages - right aligned */
    .stChatMessage[data-testid="chat-message-user"] {
        margin-left: auto !important;
        margin-right: 1rem !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 20px 20px 5px 20px !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    
    /* Assistant messages - left aligned */
    .stChatMessage[data-testid="chat-message-assistant"] {
        margin-right: auto !important;
        margin-left: 1rem !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] > div {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        border-radius: 20px 20px 20px 5px !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    
    /* Avatar styling */
    .stChatMessage .stAvatar {
        width: 45px !important;
        height: 45px !important;
        border-radius: 50% !important;
        border: 3px solid white !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    }
    
    /* User avatar */
    .stChatMessage[data-testid="chat-message-user"] .stAvatar {
        background: #2196f3 !important;
    }
    
    /* Assistant avatar */
    .stChatMessage[data-testid="chat-message-assistant"] .stAvatar {
        background: #ff5722 !important;
    }
    
    /* Chat input styling */
    .stChatInput > div > div > div > div {
        border-radius: 25px !important;
        border: 2px solid #2a5298 !important;
    }
    
    /* Improved text readability in chat messages */
    .stChatMessage[data-testid="chat-message-user"] .markdown-text-container,
    .stChatMessage[data-testid="chat-message-assistant"] .markdown-text-container {
        color: white !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] h1,
    .stChatMessage[data-testid="chat-message-user"] h2,
    .stChatMessage[data-testid="chat-message-user"] h3,
    .stChatMessage[data-testid="chat-message-assistant"] h1,
    .stChatMessage[data-testid="chat-message-assistant"] h2,
    .stChatMessage[data-testid="chat-message-assistant"] h3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Territory Financial Policy Analysis System</h1>
        <p>AI-Powered Analysis of Government Financial Strategies and Performance</p>
        <p><strong>Policy Framework:</strong> 2005-06 Budget and Forward Estimates | <strong>Compliance:</strong> Financial Management Act 1996</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    
    # Sidebar with professional layout
    with st.sidebar:
        st.markdown("## 📊 Analysis Dashboard")
        
        # Key Performance Indicators
        st.markdown("### 🎯 Key Performance Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Credit Rating", "AAA", "Maintained")
            st.metric("Budget Cycle", "Balanced", "Over Economic Cycle")
        with col2:
            st.metric("Net Interest", "-1.3%", "of Own-Source Revenue")
            st.metric("Debt Level", "Low", "Policy Maintained")
        
        st.markdown("---")
        
        # Quick Analysis Options
        st.markdown("### 🔍 Quick Analysis")
        analysis_topics = [
            "📊 Budget Performance Overview",
            "💰 Fiscal Sustainability Analysis", 
            "🏦 Debt Management Strategy",
            "🏗️ Infrastructure Investment Portfolio",
            "💼 Taxation Policy Framework",
            "📈 Economic Cycle Management",
            "🎯 Superannuation Funding Status",
            "⚖️ Financial Risk Assessment"
        ]
        
        for topic in analysis_topics:
            if st.button(topic, key=topic):
                # Extract the main topic for query
                main_topic = topic.split(" ", 1)[1]  # Remove emoji and get text
                st.session_state.quick_query = main_topic
        
        st.markdown("---")
        
        # System Information
        st.markdown("### ℹ️ System Information")
        st.info("""
        **Analysis Engine:** ChromaDB Vector Search  
        **Knowledge Base:** Territory Financial Policy Document  
        **Coverage:** 2005-06 Budget & Forward Estimates  
        **Last Updated:** Current Session
        """)
        
        # Professional Actions
        st.markdown("### 🛠️ Session Management")
        if st.button("📋 Generate Executive Summary"):
            st.session_state.show_summary = True
        
        if st.button("🔄 Reset Analysis Session"):
            st.session_state.messages = []
            st.session_state.chatbot.memory.clear_history()
            st.success("Session reset successfully")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## 💬 Financial Policy Analysis Interface")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            welcome_msg = """
## Welcome to the Territory Financial Policy Analysis System

I'm your AI assistant for analyzing the Territory's financial policies and strategies. I can provide detailed analysis on:

**Core Policy Areas:**
- Budget performance and economic cycle management
- Debt management and interest cost analysis  
- Infrastructure investment strategies
- Taxation policy and GSP relationships
- Superannuation funding progress
- Financial risk assessment and mitigation

**Analysis Capabilities:**
- Historical performance review
- Forward estimate projections
- Comparative trend analysis
- Policy compliance assessment

Please ask your question or use the Quick Analysis options in the sidebar.
            """
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
        
        # Handle quick query from sidebar
        if hasattr(st.session_state, 'quick_query'):
            query = st.session_state.quick_query
            delattr(st.session_state, 'quick_query')
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)
            
            # Get response with streaming
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing financial policy data..."):
                    response = chatbot.ask(query)
                # Stream the response
                streamed_response = stream_response(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Chat input
        if prompt := st.chat_input("Enter your financial policy analysis question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            # Get assistant response with streaming
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing financial policy data..."):
                    response = chatbot.ask(prompt)
                # Stream the response
                streamed_response = stream_response(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.markdown("## 📋 Quick Reference")
        
        # Financial Objectives Summary
        st.markdown("""
        ### 🎯 Financial Objectives
        
        **Short-term:**
        - Maintain AAA credit rating
        - Low debt levels
        - Strategic capital works
        - High service standards
        
        **Long-term:**
        - Territory infrastructure
        - Long-term liability provision
        - Financial risk minimization
        """)
        
        # Key Measures
        st.markdown("""
        ### 📊 Key Measures
        
        **Performance Indicators:**
        - Balanced budget over cycle
        - Capital infrastructure maintenance
        - Net interest cost < 0%
        - Taxation as % of GSP
        - Territory net assets growth
        - 90% superannuation funding by 2039-40
        """)
    
    # Executive Summary Modal
    if hasattr(st.session_state, 'show_summary') and st.session_state.show_summary:
        st.markdown("## 📋 Executive Summary")
        
        summary = chatbot.get_conversation_summary()
        st.markdown(f"""
        <div class="metric-card">
        <h4>Session Analysis Summary</h4>
        {summary}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Close Summary"):
            delattr(st.session_state, 'show_summary')
            st.rerun()

if __name__ == "__main__":
    main()
