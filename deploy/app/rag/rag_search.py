"""
RAG Search Module - Handles semantic search and classification using vector embeddings.
Based on the working implementation from deprecated/pre_process.py
"""

import pandas as pd
import psycopg2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from embedding.vectorization import Vectorization
from util.vect_db_conn import VectorDBConnection


class RAGSearcher:
    """
    RAG (Retrieval-Augmented Generation) Search class for semantic similarity search
    and text classification using vector embeddings.
    """
    
    def __init__(self, model_name='jhgan/ko-sbert-sts', similarity_threshold=0.24, max_workers=8):
        """
        Initialize RAG Searcher.
        
        Parameters:
        model_name (str): SentenceTransformer model name
        similarity_threshold (float): Threshold for auto-classification vs human review
        max_workers (int): Number of worker threads for parallel processing
        """
        self.vectorizer = Vectorization(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers
        self.fallback_threshold = 0.01  # For handling very similar duplicates
        
    def search_top_k(self, word: str, k: int = 3) -> Tuple[str, List[Tuple]]:
        """
        Search for top-k similar items for a single word.
        
        Parameters:
        word (str): Input word to search for
        k (int): Number of top results to return
        
        Returns:
        Tuple[str, List[Tuple]]: (word, [(classified_word, code, distance), ...])
        """
        # Create new DB connection for thread safety
        db = VectorDBConnection()
        
        try:
            # Get embedding for the word
            embedding_str = self.vectorizer.encode(word, normalize=True)
            
            # Search in database
            results = db.search_similar(embedding_str, k)
            
            if not results:
                return (word, [{
                    'word': word,
                    'classified': None,
                    'code': None,
                    'distance': None,
                    'match': False
                }])
                
            return (word, results)
            
        finally:
            db.close()
    
    def search_batch(self, words: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for multiple words using parallel processing.
        
        Parameters:
        words (List[str]): List of words to search for
        k (int): Number of top results per word
        
        Returns:
        List[Dict]: List of search results with metadata
        """
        all_results = []
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.search_top_k, word, k) for word in words]
            search_results = [future.result() for future in futures]
        
        # Process results
        for word, results in search_results:
            if results:
                for res in results:
                    all_results.append({
                        'word': word,
                        'classified': res[0],
                        'code': res[1],
                        'distance': res[2]
                    })
        
        return all_results
    
    def process_dataframe_column(self, df: pd.DataFrame, column_name: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Process all text in a specific DataFrame column.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of column to process
        k (int): Number of top results per word
        
        Returns:
        List[Dict]: Search results with column metadata
        """
        # Convert column to string and get unique words
        df[column_name] = df[column_name].astype(str)
        search_words = df[column_name].tolist()
        
        # Search for all words
        results = self.search_batch(search_words, k)
        
        # Add column information
        for result in results:
            result['column'] = column_name
            
        return results
    
    def process_multiple_columns(self, df: pd.DataFrame, columns: List[str], k: int = 3) -> pd.DataFrame:
        """
        Process multiple DataFrame columns and return consolidated results.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (List[str]): List of column names to process
        k (int): Number of top results per word
        
        Returns:
        pd.DataFrame: Consolidated search results
        """
        all_results = []
        
        for col in columns:
            col_results = self.process_dataframe_column(df, col, k)
            all_results.extend(col_results)
        
        results_df = pd.DataFrame(all_results)
        
        # Apply fallback threshold logic
        if not results_df.empty:
            results_df['distance'] = results_df.apply(
                lambda x: 1 if (x['distance'] < self.fallback_threshold and 
                               x['classified'] != x['word']) else x['distance'], 
                axis=1
            )
        
        return results_df
    
    def filter_by_threshold(self, results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter results by similarity threshold.
        
        Parameters:
        results_df (pd.DataFrame): Search results DataFrame
        
        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (auto_classified, needs_review)
        """
        if results_df.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        auto_classified = results_df[results_df['distance'] >= self.similarity_threshold].copy()
        needs_review = results_df[results_df['distance'] < self.similarity_threshold].copy()
        
        return auto_classified, needs_review
    
    def validate_against_ground_truth(self, results_df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate search results against ground truth data.
        
        Parameters:
        results_df (pd.DataFrame): Search results
        validation_df (pd.DataFrame): Ground truth validation data
        
        Returns:
        pd.DataFrame: Results with validation match information
        """
        # Merge with validation data
        merged = pd.merge(results_df, validation_df, on=['column', 'word'], how='right')
        merged['code_validate'] = merged['code_validate'].fillna(0).astype(int)
        merged['match'] = merged['code'] == merged['code_validate']
        merged = merged.drop_duplicates()
        
        return merged
    
    def get_review_candidates(self, results_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get items that need human review, sorted by confidence.
        
        Parameters:
        results_df (pd.DataFrame): Search results
        top_n (int): Number of top candidates to return
        
        Returns:
        pd.DataFrame: Items needing review sorted by distance
        """
        needs_review = results_df[results_df['distance'] < self.similarity_threshold].copy()
        return needs_review.sort_values(by='distance', ascending=True).head(top_n)
    
    def create_classification_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for classification results.
        
        Parameters:
        results_df (pd.DataFrame): Search results
        
        Returns:
        Dict: Summary statistics
        """
        if results_df.empty:
            return {
                'total_items': 0,
                'auto_classified': 0,
                'needs_review': 0,
                'auto_classification_rate': 0.0,
                'avg_confidence': 0.0
            }
        
        auto_classified = (results_df['distance'] >= self.similarity_threshold).sum()
        needs_review = (results_df['distance'] < self.similarity_threshold).sum()
        total = len(results_df)
        
        return {
            'total_items': total,
            'auto_classified': auto_classified,
            'needs_review': needs_review,
            'auto_classification_rate': auto_classified / total if total > 0 else 0.0,
            'avg_confidence': 1 - results_df['distance'].mean(),  # Convert distance to confidence
            'similarity_threshold': self.similarity_threshold
        }


class RAGBuilder:
    """
    RAG Builder class for creating and populating the RAG knowledge base
    from Excel codeframe files.
    """
    
    def __init__(self, model_name='jhgan/ko-sbert-sts'):
        """
        Initialize RAG Builder.
        
        Parameters:
        model_name (str): SentenceTransformer model name
        """
        self.vectorizer = Vectorization(model_name)
        self.db = VectorDBConnection()
    
    def parse_codeframe_excel(self, file_path: str, sheet_name: str = '공통문항_문4,5이미지') -> pd.DataFrame:
        """
        Parse codeframe Excel file and extract structured data.
        
        Parameters:
        file_path (str): Path to Excel file
        sheet_name (str): Sheet name to read
        
        Returns:
        pd.DataFrame: Parsed and structured data
        """
        # Read Excel file
        raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        raw = raw.dropna(axis=1, how='all')
        raw.columns = [*range(len(raw.columns))]
        
        # Process metadata rows
        meta = raw.iloc[:10]
        meta_cols = meta.columns.tolist()
        meta = meta.replace("NaN", pd.NA)
        
        # Extract metadata structure
        store = pd.DataFrame()
        for i in range(len(meta_cols)//2):
            for row in meta.index:
                code_col = meta_cols[i*2]
                text_col = meta_cols[i*2 + 1]
                
                meta_code = meta.iloc[row, code_col]
                meta_text = meta.iloc[row, text_col]
                
                metaname = None
                if pd.isna(meta_code) and not pd.isna(meta_text):
                    metaname = meta_text
                elif not pd.isna(meta_code) and pd.isna(meta_text):
                    metaname = meta_code
                elif not pd.isna(meta_code) and not pd.isna(meta_text):
                    metaname = f"{meta_code}_{meta_text}"
                
                if metaname:
                    store = pd.concat([store, pd.DataFrame({
                        'metaname': [metaname],
                        'row': [str(row)],
                        'column': [str(i)]
                    })], ignore_index=True)
        
        store = store.dropna(subset=['metaname'])
        store = store[store['metaname'].apply(type) == str]
        
        # Create key mappings
        first_key = store[store['row'] == '0']
        second_key = store[store['row'] == '1'] 
        third_key = store[store['row'] == '2']
        
        # Parse main data
        db_parsed = pd.DataFrame()
        for i in range(len(meta_cols)//2):
            code_col = meta_cols[i*2]
            text_col = meta_cols[i*2 + 1]
            
            meta_code = raw[code_col]
            meta_text = raw[text_col]
            
            temp = pd.DataFrame({
                'text': meta_text,
                'code': meta_code,
                'row': [*range(len(meta_text))],
                'column': [f'{int(i)}'] * len(meta_text)
            })
            
            db_parsed = pd.concat([db_parsed, temp], ignore_index=True)
        
        # Add key mappings
        db_parsed = db_parsed.dropna()
        db_parsed['first_key'] = db_parsed['column'].apply(
            lambda x: first_key[first_key['column'] == x]['metaname'].values[0] 
            if x in first_key['column'].values else None
        )
        db_parsed['second_key'] = db_parsed['column'].apply(
            lambda x: second_key[second_key['column'] == x]['metaname'].values[0] 
            if x in second_key['column'].values else None
        )
        db_parsed['third_key'] = db_parsed['column'].apply(
            lambda x: third_key[third_key['column'] == x]['metaname'].values[0] 
            if x in third_key['column'].values else None
        )
        
        # Forward fill keys
        db_parsed['first_key'] = db_parsed['first_key'].fillna(method='ffill')
        db_parsed['second_key'] = db_parsed['second_key'].fillna(method='ffill')
        db_parsed['third_key'] = db_parsed['third_key'].fillna(method='ffill')
        
        return db_parsed
    
    def split_text_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split text entries that contain multiple values separated by '/'.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        pd.DataFrame: DataFrame with split text entries
        """
        df = df.copy()
        
        # Split text on '/' separator
        df['text_split'] = df['text'].apply(
            lambda x: [t.strip() for t in str(x).split('/')] if '/' in str(x) else [x]
        )
        
        # Explode to create separate rows
        df_exploded = df.explode('text_split')
        df_exploded['text'] = df_exploded['text_split']
        df_exploded = df_exploded.drop(columns=['text_split'])
        
        return df_exploded.reset_index(drop=True)
    
    def build_rag_database(self, df: pd.DataFrame, batch_size: int = 32) -> None:
        """
        Build RAG database from processed DataFrame.
        
        Parameters:
        df (pd.DataFrame): Processed data from codeframe
        batch_size (int): Batch size for embedding generation
        """
        # Clean data
        df = df[df['text'] != ' ']
        df = df[df['text'].str.strip() != '']
        df['code'] = df['code'].astype('Int64')
        
        # Generate embeddings
        embeddings = self.vectorizer.encode_batch(
            df['text'].tolist(), 
            batch_size=batch_size, 
            normalize=True, 
            show_progress=True
        )
        
        # Add embeddings to DataFrame
        df['embedding'] = list(embeddings)
        df = df.sort_values(by='text').reset_index(drop=True)
        
        # Prepare data for database insertion
        def safe_row(row):
            def safe_val(val):
                return val if pd.notna(val) and val != '' else None
            
            emb = row['embedding']
            if isinstance(emb, np.ndarray):
                emb_str = str(emb.tolist())
            elif pd.isna(emb) or emb is None:
                emb_str = None
            else:
                emb_str = str(emb)
            
            return (
                safe_val(row['text']),
                safe_val(row['first_key']),
                safe_val(row['second_key']),
                safe_val(row['third_key']),
                int(row['code']) if pd.notna(row['code']) else None,
                emb_str
            )
        
        rows = [safe_row(row) for _, row in df.iterrows()]
        
        # Create table and insert data
        self.db.create_table()
        self.db.insert_batch(rows)
        
        print(f"Successfully inserted {len(rows)} items into RAG database")
    
    def close(self):
        """Close database connection."""
        self.db.close()
