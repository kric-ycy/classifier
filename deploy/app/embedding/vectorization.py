from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class Vectorization:
    def __init__(self, model_name='jhgan/ko-sbert-sts'):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def normalize_torch_cpu(embeddings_np):
        """
        Normalize embeddings using PyTorch for better performance.
        
        Parameters:
        embeddings_np (np.ndarray): Input embeddings
        
        Returns:
        np.ndarray: Normalized embeddings
        """
        t = torch.from_numpy(embeddings_np.astype(np.float32))
        t = t / t.norm(dim=1, keepdim=True)
        return t.numpy()

    def encode(self, texts, normalize=True, return_numpy=False):
        """
        Encode text into embeddings.
        
        Parameters:
        texts (str or list): Text to encode
        normalize (bool): Whether to normalize embeddings
        return_numpy (bool): Whether to return numpy array or string
        
        Returns:
        str or np.ndarray: Encoded embeddings
        """
        if isinstance(texts, str):
            embedding = self.model.encode([texts])[0]
        else:
            embedding = self.model.encode(texts)
        
        if normalize:
            if embedding.ndim == 1:
                embedding = self.normalize_torch_cpu(embedding.reshape(1, -1))[0]
            else:
                embedding = self.normalize_torch_cpu(embedding)
        
        if return_numpy:
            return embedding
        else:
            vector_str = str(embedding.tolist())
            return vector_str

    def encode_batch(self, texts_list, batch_size=32, normalize=True, show_progress=True):
        """
        Encode multiple texts in batches.
        
        Parameters:
        texts_list (list): List of texts to encode
        batch_size (int): Batch size for processing
        normalize (bool): Whether to normalize embeddings
        show_progress (bool): Whether to show progress bar
        
        Returns:
        np.ndarray: Encoded embeddings
        """
        embeddings = self.model.encode(
            texts_list, 
            batch_size=batch_size, 
            show_progress_bar=show_progress
        )
        
        if normalize:
            embeddings = self.normalize_torch_cpu(embeddings)
            
        return embeddings

