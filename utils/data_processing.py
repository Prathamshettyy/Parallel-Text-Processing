import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import sqlite3
import pandas as pd

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Remove unnecessary characters, emojis, and HTML tags"""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove emojis
        emoji_pattern = re.compile(
            "[" 
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def normalize_text(self, text):
        """Lowercase, tokenize, remove stopwords, and lemmatize"""
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)

# Global processor instance
processor = TextProcessor()

def clean_and_normalize_text(text):
    """Complete text cleaning and normalization pipeline"""
    cleaned = processor.clean_text(text)
    normalized = processor.normalize_text(cleaned)
    return normalized

def load_data_to_db(df, db_path='sentiment_data.db'):
    """Store DataFrame in SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql('sentiment_data', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing data in database: {e}")
        return False

def fetch_data_from_db(db_path='sentiment_data.db', limit=None):
    """Fetch data from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM sentiment_data"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return None

def get_pos_tags(text):
    """Get Part-of-Speech tags for text"""
    tokens = word_tokenize(text)
    return pos_tag(tokens)