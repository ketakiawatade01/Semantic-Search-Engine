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
