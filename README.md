# Semantic-Search-Engine
import fitz  # PyMuPDF
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import faiss
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*clean_up_tokenization_spaces.*')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
model = BertModel.from_pretrained('bert-base-uncased')

def embed(texts):
    # Ensure texts is of the expected type
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("texts must be a string or a list of strings.")
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).numpy()

def remove_figure_lines(text):
    """Remove lines that contain references to figures, tables, diagrams, etc."""
    figure_keywords = ["figure", "fig.", "table", "diagram", "chart", "image", "graph"]
    filtered_lines = []
    
    # Split the text into lines
    lines = text.splitlines()

    for line in lines:
        # Check if the line contains any of the figure-related keywords
        if not any(keyword.lower() in line.lower() for keyword in figure_keywords):
            filtered_lines.append(line)
    
    # Join the filtered lines back into a single string
    return "\n".join(filtered_lines)

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")  # Extract text as a string
        
        # Remove lines explaining figures or tables
        filtered_text = remove_figure_lines(text)
        pages.append(filtered_text)
    
    return pages

# Load the PDF dataset
pdf_path = "dataset.pdf"  # Replace with your PDF file path
pages = extract_text_from_pdf(pdf_path)

# Embed the text page by page
embeddings = [embed(page) for page in pages]

# Flatten the list of embeddings
all_embeddings = np.vstack(embeddings)

# Create FAISS index
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)

# Save the index
faiss.write_index(index, "faiss_index_new.bin")
# Save the pages to a NumPy file (optional, for reference)
np.save("pages.npy", pages)

# app.py

import streamlit as st
import faiss
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import re
import os
import warnings
from datetime import datetime  # Import datetime module


# Suppress specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*clean_up_tokenization_spaces.*')

# Set environment variable to handle OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
model = BertModel.from_pretrained('bert-base-uncased')

def embed(texts):
    # Ensure texts is of the expected type
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("texts must be a string or a list of strings.")
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).numpy()

def highlight_query(text, query):
    query_words = query.split()
    for word in query_words:
        text = re.sub(f'({re.escape(word)})', r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return text

def search_query(query, index, pages, k=5):
    query_embedding = embed(query)
    distances, indices = index.search(query_embedding, k)
    
    exact_matches = []
    relevant_results = []
    
    for i, dist in zip(indices[0], distances[0]):
        if dist == 0:  # Consider 0 distance as an exact match
            highlighted_text = highlight_query(pages[i], query)
            exact_matches.append((highlighted_text, dist))
        else:
            relevant_results.append((pages[i], dist))
    
    relevant_results = sorted(relevant_results, key=lambda x: x[1])
    
    return exact_matches, relevant_results

# Load the FAISS index and corresponding pages
index = faiss.read_index("faiss_index_new.bin")
pages = np.load("pages.npy", allow_pickle=True)

# Streamlit UI
st.title("Semantic Search with BERT and FAISS")

# Initialize session state to track whether to show other relevant results
if 'show_other_results' not in st.session_state:
    st.session_state.show_other_results = False

# User query input
query = st.text_input("Enter your search query:")

# Search button
if st.button("Search"):
    if query:
        st.session_state.show_other_results = False  # Reset when a new search is made
        exact_matches, relevant_results = search_query(query, index, pages)
        
        # Store results in session state
        st.session_state.exact_matches = exact_matches
        st.session_state.relevant_results = relevant_results

        # Display most similar result first
        if st.session_state.exact_matches:
            st.write("### Most Relevant Result:")
            st.markdown(st.session_state.exact_matches[0][0], unsafe_allow_html=True)
        
        elif st.session_state.relevant_results:
            st.write("### Most Relevant Result:")
            st.write(st.session_state.relevant_results[0][0])

# Button for other relevant results
if st.button("Show Other Relevant Results"):
    st.session_state.show_other_results = True

# Display other relevant results if the button was clicked
if st.session_state.show_other_results and 'relevant_results' in st.session_state:
    if len(st.session_state.relevant_results) > 1:
        st.write("### Other Relevant Results:")
        for rank, (result, dist) in enumerate(st.session_state.relevant_results[1:], start=2):
            st.write(f"**Rank {rank} (Distance: {dist}):**")
            st.write(result)

