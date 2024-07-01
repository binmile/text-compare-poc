from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from pathlib import Path
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text into individual words
    tokens = word_tokenize(text.lower())

    # Lemmatize the tokens to reduce words to their base form
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(lemmatized_tokens)

def load_word_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index

def get_average_embedding(text, word_embeddings):
    # Tokenize text into individual words
    tokens = word_tokenize(text)

    # Initialize an empty matrix to store embeddings of all words in the text
    embedding_matrix = np.zeros((len(tokens), 200))

    for i, token in enumerate(tokens):
        if token in word_embeddings:
            embedding_matrix[i] = word_embeddings[token]

    # Calculate the average embedding across all words in the text
    avg_embedding = np.mean(embedding_matrix, axis=0)
    
    return avg_embedding


def calculate_similarity(paragraph1, paragraph2, n):
    # Convert the paragraphs to lowercase
    paragraph1 = paragraph1.lower()
    paragraph2 = paragraph2.lower()
    preprocessed_paragraph1 = preprocess_text(paragraph1)
    preprocessed_paragraph2 = preprocess_text(paragraph2)

    # Generate N-grams for both paragraphs
    ngrams_paragraph1 = set(ngrams(preprocessed_paragraph1.split(), n))
    ngrams_paragraph2 = set(ngrams(preprocessed_paragraph2.split(), n))

    # Calculate Jaccard similarity coefficient
    intersection = len(ngrams_paragraph1.intersection(ngrams_paragraph2))
    union = len(ngrams_paragraph1) + len(ngrams_paragraph2) - intersection
    similarity_score = intersection / union

    return similarity_score

def calculate_cosine_similarity(paragraph1, paragraph2):
    # Preprocess paragraphs by converting to lowercase and applying lemmatization
    preprocessed_paragraph1 = preprocess_text(paragraph1)
    preprocessed_paragraph2 = preprocess_text(paragraph2)

    # Use CountVectorizer to convert preprocessed paragraphs into vectors 
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    vectorized_paragraphs = vectorizer.fit_transform([preprocessed_paragraph1, preprocessed_paragraph2])

    # Compute cosine similarity between the vectors
    cos_similarities = cosine_similarity(vectorized_paragraphs)[0][1]

    return cos_similarities

def calculate_similarity_with_glove(paragraph1, paragraph2, word_embeddings):
    # Preprocess paragraphs by converting to lowercase and applying lemmatization
    preprocessed_paragraph1 = preprocess_text(paragraph1)
    preprocessed_paragraph2 = preprocess_text(paragraph2)

    vector1_avg_embedding= get_average_embedding(preprocessed_paragraph1, word_embeddings)
    vector2_avg_embedding= get_average_embedding(preprocessed_paragraph2, word_embeddings)

    # Compute cosine similarity between the average word embeddings
    cos_similarities = cosine_similarity(vector1_avg_embedding.reshape(1, -1), vector2_avg_embedding.reshape(1, -1))

    return cos_similarities[0][0]


# Example usage:
paragraph_1 = """1: The serviceability of a business operating for more than two years in Virgin Bank is calculated based on the following criteria:
2: - The most recent personal and business tax return
3: - Financial statements and any interim financial statements
4: - ATO Notice of Assessment (for individuals)
5: - Last two years' Tax Assessment Notices and tax returns (for employed borrowers or guarantors)
6: - For top-ups, the most recent personal tax assessment notice and personal tax return
7: - For variations or switches, the most recent lodged personal tax return
8: - For non-individuals (partnership, company, or trust), the last two years' business tax returns and one set of business financial statements (accountant prepared Profit and Loss Statement and Balance Sheet) from the most recent financial year
9: - For top-ups, variations, or switches for non-individuals, the most recent lodged business tax return."""

paragraph_2 = "The serviceability of a business operating for more than two years in Virgin Bank is calculated based on the following criteria: - The most recent personal and business tax return - Financial statements and any interim financial statements - ATO Notice of Assessment (for individuals) - Last two years' Tax Assessment Notices and tax returns (for employed borrowers or guarantors) - For top-ups, the most recent personal tax assessment notice and personal tax return - For variations or switches, the most recent lodged personal tax return - For non-individuals (partnership, company, or trust), the last two years' business tax returns and one set of business financial statements (accountant prepared Profit and Loss Statement and Balance Sheet) from the most recent financial year - For top-ups, variations, or switches for non-individuals, the most recent lodged business tax return. This answer is from above information"

similarity_score_1gram = calculate_similarity(paragraph_1, paragraph_2, 1)
print("Similarity score (1-gram):", similarity_score_1gram)

similarity_score_2gram = calculate_similarity(paragraph_1, paragraph_2, 2)
print("Similarity score (2-gram):", similarity_score_2gram)

similarity_score_3gram = calculate_similarity(paragraph_1, paragraph_2, 3)
print("Similarity score (3-gram):", similarity_score_3gram)

similarity_score_4gram = calculate_similarity(paragraph_1 , paragraph_2 ,4 )
print(f"Similarity score (4-gram): {similarity_score_4gram}")

similarity_score_cosine = calculate_cosine_similarity(paragraph_1 , paragraph_2)
print(f"Similarity score (Cosine): {similarity_score_cosine}")

word_embeddings_path = Path(str(Path().resolve()) + "/data/glove.6B.200d.txt")
word_embeddings = load_word_embeddings(word_embeddings_path)

similarity_score_glove = calculate_similarity_with_glove(paragraph_1, paragraph_2, word_embeddings)
print("Similarity score (Glove):", similarity_score_glove)