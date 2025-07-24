# %%
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from typing import List, Tuple, Optional


class VectorDBConnection:
    def __init__(self, dbname='net_classifier', user='aihc', password='aihc8520', host='116.125.140.82', port='5432'):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cur = self.conn.cursor()
        # Set search path for vector operations
        self.cur.execute("SET search_path to test_schema, public;")
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.cur.close()
        self.conn.close()
        
    def insert_item(self, word: str, embedding: str, code: Optional[int] = None, 
                   first_key: Optional[str] = None, second_key: Optional[str] = None, 
                   third_key: Optional[str] = None):
        """
        Insert a single item into the database.
        
        Parameters:
        word (str): The text/word to store
        embedding (str): Vector embedding as string
        code (int, optional): Classification code
        first_key (str, optional): First classification key
        second_key (str, optional): Second classification key
        third_key (str, optional): Third classification key
        """
        query = """
        INSERT INTO test_schema.embed_test (word, embedding, code, first_key, second_key, third_key) 
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.cur.execute(query, (word, embedding, code, first_key, second_key, third_key))
        self.conn.commit()

    def insert_batch(self, items: List[Tuple]):
        """
        Insert multiple items in batch.
        
        Parameters:
        items (List[Tuple]): List of tuples containing (word, first_key, second_key, third_key, code, embedding)
        """
        insert_query = """
        INSERT INTO test_schema.embed_test (
            word, first_key, second_key, third_key, code, embedding
        ) VALUES %s
        """
        execute_values(self.cur, insert_query, items)
        self.conn.commit()

    def create_table(self):
        """Create the embed_test table if it doesn't exist."""
        self.cur.execute("DROP TABLE IF EXISTS test_schema.embed_test;")
        self.conn.commit()
        
        create_query = """
        CREATE TABLE IF NOT EXISTS test_schema.embed_test (
            id serial PRIMARY KEY, 
            word text, 
            first_key text,
            second_key text,
            third_key text,
            code integer,
            embedding public.vector(768)
        )
        """
        self.cur.execute(create_query)
        self.conn.commit()

    def search_similar(self, embedding_str: str, k: int = 3) -> List[Tuple]:
        """
        Search for similar embeddings in the database.
        
        Parameters:
        embedding_str (str): Query embedding as string
        k (int): Number of top results to return
        
        Returns:
        List[Tuple]: List of (word, code, distance) tuples
        """
        query = """
        SELECT word, code, embedding <=> %s::vector AS distance
        FROM test_schema.embed_test
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        self.cur.execute(query, (embedding_str, embedding_str, k))
        results = self.cur.fetchall()
        return results if results else []
