# memory/case_memory.py
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import OpenAI

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Install with: pip install chromadb")

class CaseMemory:
    """
    Stores past cases and retrieves relevant ones using similarity or learned value.
    Uses ChromaDB for efficient vector storage and retrieval with fallback to in-memory.
    """
    
    def __init__(self, api_key=None, 
                 memory_dir: str = "memory_db", 
                 collection_name: str = "cases",
                 max_cases: int = 1000):
        """
        Initialize case memory system.
        
        :param api_key: OpenAI API key, if None will read from environment
        :param memory_dir: Directory to store the memory database
        :param collection_name: Name of the ChromaDB collection
        :param max_cases: Maximum number of cases to store
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.memory_dir = memory_dir
        self.collection_name = collection_name
        self.max_cases = max_cases
        
        # Initialize storage
        self.cases = []  # Fallback storage
        self.case_counter = 0
        self.use_chromadb = False
        
        if CHROMADB_AVAILABLE:
            try:
                self._init_chromadb()
                self.use_chromadb = True
            except Exception as e:
                print(f"ChromaDB initialization failed: {e}")
                print("Using in-memory storage")
        else:
            print("Using in-memory storage for cases")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        # Create memory directory if it doesn't exist
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.memory_dir)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection '{self.collection_name}' with {self.collection.count()} cases")
        except Exception:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            print(f"Created new collection '{self.collection_name}'")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a dummy embedding of appropriate size
            return [0.0] * 1536  # text-embedding-3-small has 1536 dimensions

    def retrieve(self, task_description: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar cases for the given task.
        
        :param task_description: Description of the current task
        :param top_k: Number of similar cases to retrieve
        :return: List of similar cases
        """
        if self.use_chromadb:
            return self._retrieve_with_chromadb(task_description, top_k)
        else:
            return self._retrieve_fallback(task_description, top_k)
    
    def _retrieve_with_chromadb(self, task_description: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve cases using ChromaDB."""
        try:
            # Check if collection has any cases
            collection_count = self.collection.count()
            if collection_count == 0:
                return []
            
            # Get embedding for the task
            task_embedding = self._get_embedding(task_description)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[task_embedding],
                n_results=min(top_k, collection_count),
                include=['metadatas', 'documents', 'distances']
            )
            
            # Convert results to the expected format
            retrieved_cases = []
            if results.get('metadatas') and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    steps_str = metadata.get('steps', '[]')
                    try:
                        steps = json.loads(steps_str) if isinstance(steps_str, str) else []
                    except json.JSONDecodeError:
                        steps = []
                    
                    case = {
                        'task': metadata.get('task', ''),
                        'answer': metadata.get('answer', ''),
                        'reward': float(metadata.get('reward', 0.0)),
                        'summary': results['documents'][0][i] if results.get('documents') else '',
                        'similarity': 1 - results['distances'][0][i] if results.get('distances') else 0.0,
                        'timestamp': metadata.get('timestamp', ''),
                        'steps': steps
                    }
                    retrieved_cases.append(case)
            
            return retrieved_cases
            
        except Exception as e:
            print(f"Error retrieving from ChromaDB: {e}")
            return []
    
    def _retrieve_fallback(self, task_description: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback retrieval using in-memory storage."""
        if not self.cases:
            return []
        
        try:
            task_vec = self._get_embedding(task_description)
            # Calculate similarity with stored case embeddings
            sims = [self._cosine_sim(task_vec, case['embedding']) for case in self.cases]
            # Take top-k by similarity
            top_indices = np.argsort(sims)[::-1][:top_k]
            top_cases = []
            
            for i in top_indices:
                case = self.cases[i].copy()
                case['similarity'] = sims[i]
                # Remove embedding from returned case to save space
                if 'embedding' in case:
                    del case['embedding']
                top_cases.append(case)
            
            return top_cases
            
        except Exception as e:
            print(f"Error in fallback retrieval: {e}")
            return []

    def store_case(self, task_description: str, final_answer: str, 
                   reward: float, steps_log: Optional[List] = None) -> bool:
        """
        Add a new case to memory after completing a task.
        
        :param task_description: Description of the task
        :param final_answer: The final answer/solution
        :param reward: Reward/score for this case
        :param steps_log: Optional log of steps taken
        :return: True if successfully stored
        """
        if self.use_chromadb:
            return self._store_with_chromadb(task_description, final_answer, reward, steps_log)
        else:
            return self._store_fallback(task_description, final_answer, reward, steps_log)
    
    def _store_with_chromadb(self, task_description: str, final_answer: str, 
                           reward: float, steps_log: Optional[List] = None) -> bool:
        """Store case using ChromaDB."""
        try:
            # Get embedding for the task
            case_embedding = self._get_embedding(task_description)
            
            # Create case summary
            summary = f"Task: {task_description} | Outcome: {final_answer} | Reward: {reward:.2f}"
            
            # Prepare metadata
            metadata = {
                'task': task_description,
                'answer': final_answer,
                'reward': str(reward),
                'timestamp': datetime.now().isoformat(),
                'steps': json.dumps(steps_log or [])
            }
            
            # Generate unique ID
            case_id = f"case_{datetime.now().timestamp()}_{hash(task_description) % 10000}"
            
            # Add to collection
            self.collection.add(
                embeddings=[case_embedding],
                documents=[summary],
                metadatas=[metadata],
                ids=[case_id]
            )
            
            # Clean up old cases if needed
            if self.collection.count() > self.max_cases:
                self._cleanup_old_cases()
            
            return True
            
        except Exception as e:
            print(f"Error storing case in ChromaDB: {e}")
            return False
    
    def _store_fallback(self, task_description: str, final_answer: str, 
                       reward: float, steps_log: Optional[List] = None) -> bool:
        """Fallback storage using in-memory storage."""
        try:
            case_embedding = self._get_embedding(task_description)
            summary = f"Task: {task_description} | Outcome: {final_answer} | Reward: {reward:.2f}"
            
            new_case = {
                'task': task_description,
                'answer': final_answer,
                'reward': reward,
                'embedding': case_embedding,
                'summary': summary,
                'steps': steps_log or [],
                'timestamp': datetime.now().isoformat()
            }
            
            self.cases.append(new_case)
            
            # Clean up old cases if needed
            if len(self.cases) > self.max_cases:
                self.cases = self.cases[-self.max_cases:]
            
            return True
            
        except Exception as e:
            print(f"Error storing case in fallback: {e}")
            return False
    
    def _cleanup_old_cases(self):
        """Remove oldest cases from ChromaDB collection."""
        try:
            # Get all cases with metadata
            all_cases = self.collection.get(include=['metadatas'])
            
            if all_cases.get('ids') and len(all_cases['ids']) > self.max_cases:
                # Sort by timestamp and remove oldest
                cases_with_time = []
                metadatas = all_cases.get('metadatas', [])
                
                for i, case_id in enumerate(all_cases['ids']):
                    if i < len(metadatas) and metadatas[i]:
                        timestamp = metadatas[i].get('timestamp', '')
                        cases_with_time.append((timestamp, case_id))
                
                if cases_with_time:
                    cases_with_time.sort()  # Sort by timestamp
                    
                    # Remove oldest cases
                    cases_to_remove = len(all_cases['ids']) - self.max_cases
                    ids_to_remove = [case_id for _, case_id in cases_with_time[:cases_to_remove]]
                    
                    self.collection.delete(ids=ids_to_remove)
                    print(f"Cleaned up {cases_to_remove} old cases")
                
        except Exception as e:
            print(f"Error cleaning up old cases: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        if self.use_chromadb:
            try:
                count = self.collection.count()
            except Exception:
                count = 0
        else:
            count = len(self.cases)
        
        return {
            'total_cases': count,
            'max_cases': self.max_cases,
            'storage_type': 'ChromaDB' if self.use_chromadb else 'In-memory',
            'memory_dir': self.memory_dir if self.use_chromadb else None
        }
    
    def clear_memory(self) -> bool:
        """Clear all stored cases."""
        try:
            if self.use_chromadb:
                # Delete the collection and recreate it
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
            else:
                self.cases = []
                self.case_counter = 0
            
            print("Memory cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False
    
    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
        except Exception:
            return 0.0
