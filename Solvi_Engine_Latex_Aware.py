import os
import shutil
import random
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

CHROMA_PATH = "chroma_Jabran"
DATA_PATH = "DataNougatParsed"
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'  # Sentence-BERT model
MAX_TOKENS_PER_CHUNK = 700
TARGET_AVG_TOKENS_PER_CHUNK = 500

def custom_tokenize(text):
    equation_pattern = r'\$[^$]+\$|\\\[[^\]]+\\\]'
    parts = re.split(equation_pattern, text)
    tokens = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            tokens.extend(word_tokenize(part))
        else:
            tokens.append(part)
    return tokens

def load_documents(file_path):
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents

def generate_sentence_embeddings(sentences, model):
    def preprocess_for_embedding(sentence):
        return re.sub(r'\$[^$]+\$|\\\[[^\]]+\\\]', '[EQUATION]', sentence)
    
    preprocessed_sentences = [preprocess_for_embedding(s) for s in sentences]
    return model.encode(preprocessed_sentences)

def calculate_gap_scores(embeddings, n=3):
    gap_scores = []
    for i in range(len(embeddings) - n):
        similarity = cosine_similarity(embeddings[i:i+n], embeddings[i+n:i+2*n])
        gap_scores.append(np.mean(similarity))
    return gap_scores

def smooth_gap_scores(gap_scores, k=5):
    return np.convolve(gap_scores, np.ones(k)/k, mode='valid')

def detect_boundaries(smoothed_gap_scores, c=1.2):
    local_minima = (np.diff(np.sign(np.diff(smoothed_gap_scores))) > 0).nonzero()[0] + 1
    significant_boundaries = [i for i in local_minima if smoothed_gap_scores[i] < np.mean(smoothed_gap_scores) - c * np.std(smoothed_gap_scores)]
    return significant_boundaries

def chunk_text_with_embeddings(sentences, embeddings, boundaries, max_tokens=MAX_TOKENS_PER_CHUNK, target_avg_tokens=TARGET_AVG_TOKENS_PER_CHUNK):
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = custom_tokenize(sentence)
        sentence_length = len(sentence_tokens)

        if current_chunk_tokens + sentence_length > max_tokens or (len(current_chunk) > 0 and i in boundaries):
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_tokens = sentence_length
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def split_text(documents: list[Document]):
    all_chunks = []
    total_tokens = 0
    num_chunks = 0
    model = SentenceTransformer(MODEL_NAME)

    for document in documents:
        text = document.page_content
        sentences = sent_tokenize(text)
        embeddings = generate_sentence_embeddings(sentences, model)
        gap_scores = calculate_gap_scores(embeddings)
        smoothed_gap_scores = smooth_gap_scores(gap_scores)
        boundaries = detect_boundaries(smoothed_gap_scores)
        document_chunks = chunk_text_with_embeddings(sentences, embeddings, boundaries)
        num_chunks += len(document_chunks)
        total_tokens += sum(len(custom_tokenize(chunk)) for chunk in document_chunks)

        for chunk in document_chunks:
            new_doc = Document(page_content=chunk, metadata=document.metadata)
            all_chunks.append(new_doc)

    avg_tokens_per_chunk = total_tokens / num_chunks if num_chunks else 0
    print(f"Split {len(documents)} documents into {num_chunks} chunks.")
    print(f"Average tokens per chunk: {avg_tokens_per_chunk}")
    return all_chunks, avg_tokens_per_chunk

def generate_data_store():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

    all_chunks = []

    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".mmd"):
            file_path = os.path.join(DATA_PATH, filename)
            documents = load_documents(file_path)
            chunks, avg_tokens_per_chunk = split_text(documents)
            all_chunks.extend(chunks)
            db.add_documents(chunks)
            print(f"Processed and added {filename} to the database.")
            print(f"Average tokens per chunk for {filename}: {avg_tokens_per_chunk}")

    db.persist()
    print(f"All documents have been processed and saved to {CHROMA_PATH}.")
    return all_chunks

def print_random_chunks(chunks, num_chunks=2):
    if len(chunks) > num_chunks:
        selected_chunks = random.sample(chunks, num_chunks)
    else:
        selected_chunks = chunks

    for i, chunk in enumerate(selected_chunks):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")

def main():
    all_chunks = generate_data_store()
    print_random_chunks(all_chunks)

if __name__ == "__main__":
    main()