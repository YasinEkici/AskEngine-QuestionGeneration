import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import re
from io import StringIO
from app.utils.pdf_cleaner import repair_pdf_text

# Page Config (Must be first)
st.set_page_config(
    page_title="AskEngine | AI Question Generation",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
    }
    .stButton > button {
        border-radius: 20px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .history-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 5px solid #ff4b4b;
        color: var(--text-color);
    }
    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
col_header_1, col_header_2 = st.columns([1, 4])
with col_header_1:
    st.markdown("# ✨ AskEngine")
with col_header_2:
    st.markdown("### Advanced Question Generation System")
    st.caption("Powered by T5-Large & RTX 5080")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_status = st.empty()
    try:
        if requests.get("http://127.0.0.1:8000/health", timeout=1).status_code == 200:
            api_status.success("API Online")
        else:
            api_status.error("API Offline")
    except:
        api_status.error("API Offline")
    
    st.markdown("### Model Settings")
    with st.expander("🔧 Advanced Settings"):
        beam_size = st.slider("Beam Size", 1, 10, 4, help="Higher values explore more possibilities")
        max_length = st.number_input("Max Length", 32, 128, 64)
        num_candidates = st.slider("Candidates", 1, 5, 3, help="Number of alternative questions to generate")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("AskEngine uses a fine-tuned Transformer model to generate relevant questions from context.")

# Main Interface
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("📝 Input Context")

    # Define Examples
    example_options = {
        "Custom": "",
        "Science (ML)": "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.",
        "History (Apollo 11)": "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC.",
        "Geography (Amazon)": "The Amazon River in South America is the largest river by discharge volume of water in the world, and by some definitions it is the longest."
    }
    
    selected_example = st.selectbox("Choose an example or type your own:", list(example_options.keys()))
    default_context = example_options[selected_example]
    
    default_answer = ""
    if selected_example == "Science (ML)": default_answer = "Machine learning"
    elif selected_example == "History (Apollo 11)": default_answer = "Neil Armstrong"
    elif selected_example == "Geography (Amazon)": default_answer = "Amazon River"

    # Initialize session state for context if not present
    if "context_text" not in st.session_state:
        st.session_state.context_text = default_context
        st.session_state.full_text_source = None # To store huge PDFs
    
    # Update session state if example selection changes (and it's not Custom)
    if selected_example != "Custom" and st.session_state.context_text != default_context:
         st.session_state.context_text = default_context
         st.session_state.full_text_source = None

    # File Upload Section
    with st.expander("📂 Upload Document (PDF/TXT)", expanded=True):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                import pypdf
                try:
                    pdf_reader = pypdf.PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Repair text layout
                    text = repair_pdf_text(text)
                    
                    st.session_state.full_text_source = text
                    # Show preview only to capture input
                    preview_len = 1000
                    st.session_state.context_text = text[:preview_len] + f"\n\n... [Total {len(text)} characters loaded. Context will be auto-extracted based on answer]"
                    st.success(f"PDF Loaded: {len(pdf_reader.pages)} pages. (Large files are handled automatically)")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
            else:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                text = stringio.read()
                st.session_state.full_text_source = text
                st.session_state.context_text = text[:1000] + "\n..." if len(text) > 1000 else text
                st.success("Text file loaded.")
    
    # Text Area
    context_display = st.text_area(
        "Context Paragraph (Preview)", 
        value=st.session_state.context_text, 
        height=250, 
        placeholder="Paste your text here or upload a file...",
        disabled=st.session_state.full_text_source is not None # Read-only if file loaded to avoid confusion
    )
    
    # Allow manual override (Clear file)
    if st.session_state.full_text_source is not None:
        if st.button("Clear File & Reset"):
            st.session_state.full_text_source = None
            st.session_state.context_text = ""
            st.rerun()

    # Sync text area back if manual entry (and no file loaded)
    if st.session_state.full_text_source is None:
        st.session_state.context_text = context_display

    # Answer Suggestion
    suggestions = []
    if st.button("🪄 Auto-Suggest Answers"):
        with st.spinner("Finding key phrases..."):
            try:
                # Use current context
                ctx = st.session_state.full_text_source if st.session_state.full_text_source else st.session_state.context_text
                if ctx and len(ctx) > 10:
                     res = requests.post("http://127.0.0.1:8000/suggest_answers", json={"context": ctx, "max_suggestions": 8})
                     if res.status_code == 200:
                         st.session_state.suggestions = res.json()["candidates"]
                     else:
                         st.warning("Could not fetch suggestions.")
                else:
                    st.warning("Please provide context first.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Display Suggestions as Pills (Streamlit 1.40+ has st.pills, but fallback to buttons for safety if version unknowns)
    # Using columns for a pill-like effect
    if "suggestions" in st.session_state and st.session_state.suggestions:
        st.markdown("**/ Suggested Answers:**")
        # Use simple st.button loop. When clicked, it sets session state 'answer_input'
        cols = st.columns(4)
        for i, cand in enumerate(st.session_state.suggestions):
            if cols[i % 4].button(cand, key=f"sugg_{i}", use_container_width=True):
                 st.session_state.answer_input = cand
                 st.rerun()

    answer = st.text_input("Target Answer", value=default_answer, key="answer_input", placeholder="The answer you want to target...")
    
    generate_btn = st.button("🚀 Generate Question", type="primary")

def extract_context_window(full_text, target_answer, window_size_words=250):
    """
    Finds the answer in the text and returns a window of WORDS around it.
    Acts as a proxy for token-based windowing (1 word ~ 1.3 tokens).
    target: ~350-400 tokens -> ~250-300 words.
    """
    if not target_answer or not full_text:
        return full_text[:window_size_words * 5] # Fallback (approx chars)
        
    # Normalized search
    text_lower = full_text.lower()
    answer_lower = target_answer.lower()
    
    start_char_idx = text_lower.find(answer_lower)
    if start_char_idx == -1:
        return None # Answer not found
    
    # Calculate word indices
    # We slice the text up to the answer to count words before it
    pre_answer_text = full_text[:start_char_idx]
    words_before = pre_answer_text.split()
    target_word_idx = len(words_before)
    
    all_words = full_text.split()
    
    # Define window
    half_window = window_size_words // 2
    start_word = max(0, target_word_idx - half_window)
    end_word = min(len(all_words), target_word_idx + half_window + len(answer_lower.split()))
    
    # Reconstruct
    extracted_words = all_words[start_word:end_word]
    return " ".join(extracted_words)

with col2:
    st.subheader("💡 Results")
    
    # --- GENERATION LOGIC ---
    if generate_btn:
        if not answer:
            st.warning("Please provide a Target Answer.")
        else:
            # 1. Determine Context
            final_context = st.session_state.context_text
            
            if st.session_state.full_text_source:
                with st.spinner("🔍 Smart Chunking: Finding relevant window..."):
                    extracted = extract_context_window(st.session_state.full_text_source, answer)
                    if extracted:
                        final_context = extracted
                        st.caption(f"✅ Found answer context ({len(final_context)} chars).")
                    else:
                        st.warning("Answer not found in document. Using start of text.")
                        final_context = st.session_state.full_text_source[:1000]

            # 2. Call API
            with st.spinner("🧠 Generating Questions..."):
                try:
                    payload = {
                        "context": final_context, 
                        "answer": answer,
                        "beam_size": beam_size,
                        "max_length": max_length,
                        "num_candidates": num_candidates
                    }
                    response = requests.post("http://127.0.0.1:8000/generate_question", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Enrich data
                        data["answer"] = answer
                        data["context"] = final_context
                        data["timestamp"] = datetime.now().strftime("%H:%M:%S")
                        
                        # Add to history
                        st.session_state.history.insert(0, data)
                        # st.rerun() # Optional: Force refresh to ensure state consistency
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

    # --- DISPLAY LOGIC (State-Based) ---
    if st.session_state.history:
        latest = st.session_state.history[0]
        
        # Result Card
        with st.container():
            # Main Question
            st.markdown(f"### {latest.get('question', 'Error')}")
            
            # Alternatives Pill Box
            candidates = latest.get("candidates", [])
            if len(candidates) > 1:
                st.markdown("---")
                st.markdown("**🔄 Alternatives:**")
                for cand in candidates:
                     if cand != latest.get('question'):
                         st.info(cand)

            # Context Expander
            with st.expander("🔎 View Focused Source Context"):
                # Span Highlight Logic
                try:
                    escaped_ans = re.escape(latest['answer'])
                    highlighted = re.sub(
                        f"({escaped_ans})", 
                        r'<mark style="background-color: #ffd700; color: black; font-weight: bold; border-radius: 4px; padding: 0 2px;">\1</mark>', 
                        latest['context'], 
                        flags=re.IGNORECASE
                    )
                    st.markdown(highlighted, unsafe_allow_html=True)
                except:
                    st.write(latest['context'])

            # Inline Feedback Form
            st.markdown("---")
            with st.container():
                st.markdown("#### ✍️ Rate & Improve")
                f1, f2 = st.columns([3, 1])
                with f1:
                    # Unique key ensures inputs don't clash between generations
                    uid = latest.get("timestamp") + str(len(st.session_state.history))
                    corrected_q = st.text_input("Refine Question", value=latest['question'], key=f"fix_{uid}")
                with f2:
                    rating_val = st.slider("Score", 1, 5, 4, key=f"rate_{uid}")
                
                if st.button("💾 Save Feedback", key=f"save_{uid}"):
                    try:
                        fb_payload = {
                            "context": latest['context'], 
                            "answer": latest['answer'],
                            "generated_question": latest['question'],
                            "corrected_question": corrected_q,
                            "rating": rating_val,
                            "model_name": latest.get("model_name", "unknown")
                        }
                        requests.post("http://127.0.0.1:8000/feedback", json=fb_payload)
                        st.toast("Feedback Saved! Thank you for training AskEngine.", icon="🎉")
                    except Exception as e:
                        st.error(f"Failed to save feedback: {e}")

    # History Expander (Collapsed)
    if len(st.session_state.history) > 1:
        st.markdown("---")
        with st.expander(f"📚 History ({len(st.session_state.history)-1} older items)"):
            for i, item in enumerate(st.session_state.history[1:]):
                st.markdown(f"**{item['timestamp']}** | Ans: *{item['answer']}*")
                st.write(f"Q: {item['question']}")
                st.markdown("---")
