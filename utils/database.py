import sqlite3
import pandas as pd
from datetime import datetime

class SentimentDatabase:
    def __init__(self, db_path='sentiment_analysis.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create texts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT,
                cleaned_text TEXT,
                true_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER,
                method TEXT,
                predicted_label TEXT,
                confidence REAL,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (text_id) REFERENCES texts (id)
            )
        ''')
        
        # Create processing_runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                method TEXT,
                num_samples INTEGER,
                total_time REAL,
                accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_texts(self, df):
        """Store texts in database"""
        conn = sqlite3.connect(self.db_path)
        df.to_sql('texts', conn, if_exists='append', index=False)
        conn.close()
    
    def store_results(self, results_df):
        """Store analysis results in database"""
        conn = sqlite3.connect(self.db_path)
        results_df.to_sql('results', conn, if_exists='append', index=False)
        conn.close()
    
    def get_processing_history(self):
        """Get history of processing runs"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM processing_runs ORDER BY created_at DESC LIMIT 10",
            conn
        )
        conn.close()
        return df
    
    def get_texts(self, limit=100):
        """Retrieve texts from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM texts LIMIT {limit}",
            conn
        )
        conn.close()
        return df