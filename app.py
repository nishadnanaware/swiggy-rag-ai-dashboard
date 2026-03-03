import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

st.set_page_config(page_title="Swiggy RAG QA", layout="wide")

# ==========================================
# SWIGGY BRAND UI THEME
# ==========================================
st.markdown("""
<style>

/* Swiggy Gradient Background */
.stApp {
    background: linear-gradient(135deg, #fc8019, #ffb347);
}

/* Main Title */
h1 {
    font-size: 42px !important;
    text-align: center;
    font-weight: 800;
    color: white !important;
}

/* Subtitle */
p {
    font-size: 18px;
    color: #2c2c2c;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.85);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    margin-bottom: 25px;
}

/* Input Styling */
input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* Metric Styling */
[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER WITH LOGO
# ==========================================
col1, col2 = st.columns([1,4])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/13/Swiggy_logo.png", width=120)

with col2:
    st.title("Swiggy Annual Report AI Dashboard")
    st.write("Ask any question related to Swiggy Annual Report 2023-24")

st.image("swiggy_rag_projectbanner.jpg", use_container_width=True)

# ----------------------------
# LOAD PDF
# ----------------------------
@st.cache_data
def load_pdf():
    loader = PyPDFLoader("Swiggy_Annual_Report.pdf")
    return loader.load()

documents = load_pdf()

# ----------------------------
# SPLIT INTO LARGE CHUNKS
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

docs = text_splitter.split_documents(documents)

# ----------------------------
# EMBEDDINGS + VECTOR STORE
# ----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

# ----------------------------
# LOAD LLM
# ----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small",
        torch_dtype=torch.float32
    )
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# USER QUERY
# ----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
query = st.text_input("🔎 Enter your question:")
st.markdown('</div>', unsafe_allow_html=True)

if query:

    retrieved_docs = retriever.invoke(query)
    combined_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    lower_query = query.lower()

    # ======================================================
    # FINANCIAL MODE
    # ======================================================
    if any(word in lower_query for word in ["revenue", "profit", "loss", "income", "expenses"]):

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("## 📊 Financial Data (FY 2023-24)")
        st.markdown('</div>', unsafe_allow_html=True)

        lines = combined_text.split("\n")
        found = False

        for line in lines:
            lower_line = line.lower()

            if "revenue" in lower_query or "income" in lower_query:
                if "total income" in lower_line:
                    numbers = re.findall(r"\(?-?\d[\d,]*\)?", line)
                    if numbers:
                        value = numbers[0].replace("(", "-").replace(")", "")
                        st.metric("💰 Total Income (Revenue)", value)
                        found = True
                        break

            if "profit" in lower_query or "loss" in lower_query:
                if "net profit" in lower_line:
                    numbers = re.findall(r"\(?-?\d[\d,]*\)?", line)
                    if numbers:
                        value = numbers[0].replace("(", "-").replace(")", "")
                        st.metric("📈 Net Profit / Loss", value)
                        found = True
                        break

            if "expenses" in lower_query:
                if "total expenses" in lower_line:
                    numbers = re.findall(r"\(?-?\d[\d,]*\)?", line)
                    if numbers:
                        value = numbers[0].replace("(", "-").replace(")", "")
                        st.metric("💸 Total Expenses", value)
                        found = True
                        break

        if not found:
            st.warning("❌ Could not clearly extract the requested financial metric.")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("## 📌 Supporting Context")
        st.write(combined_text[:1000])
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================================================
    # NORMAL RAG MODE
    # ======================================================
    else:

        prompt = f"""
Answer the question strictly using the context below.
If answer is not found, say:
"I could not find the answer in the document."

Context:
{combined_text}

Question:
{query}

Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("## 🤖 AI Generated Answer")
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("## 📌 Supporting Context")
        st.write(combined_text[:1000])
        st.markdown('</div>', unsafe_allow_html=True)