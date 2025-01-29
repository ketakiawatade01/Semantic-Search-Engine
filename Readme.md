# Semantic Search Engine with BERT and FAISS

This project is a semantic search engine that uses BERT for text embeddings and FAISS for efficient similarity search. It allows users to search through a dataset of PDF documents and retrieve the most relevant results based on their queries. The user interface is built using Streamlit.

## Features

- **Text Extraction**: Extract text from PDF files while removing lines with references to figures, tables, and other visual elements.
- **Text Embedding**: Generate embeddings for text pages using BERT.
- **Similarity Search**: Use FAISS to index and search text embeddings for similar content.
- **Streamlit Interface**: Provides an interactive UI to input search queries and display results.
