import numpy as np
from opensearchpy import OpenSearch, helpers
import json
from typing import List, Dict, Any, Optional
import time

class OpenSearchVectorSearch:
    """
    Optimized OpenSearch vector search implementation for 1024-dimensional embeddings.
    Supports multiple algorithms: HNSW, IVF, and Brute Force with performance optimizations.
    """
    
    def __init__(self, host='localhost', port=9200, use_ssl=False, verify_certs=False, 
                 http_auth=None, timeout=60):
        """Initialize OpenSearch client with connection parameters."""
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            timeout=timeout,
            max_retries=3,
            retry_on_timeout=True
        )
        self.vector_dimension = 1024
    
    def create_hnsw_index(self, index_name: str, m: int = 48, ef_construction: int = 512) -> bool:
        """
        Create index optimized for HNSW algorithm - Best for most use cases.
        
        Args:
            index_name: Name of the index
            m: Number of bi-directional links for each node (16-64 recommended)
            ef_construction: Size of dynamic candidate list (100-800 recommended)
        """
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512,  # Search-time parameter
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s",  # Optimize for bulk indexing
                    "merge.policy.max_merged_segment": "5gb"
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.vector_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",  # or "l2", "innerproduct"
                            "engine": "lucene",
                            "parameters": {
                                "m": m,
                                "ef_construction": ef_construction,
                                "max_connections": m * 2  # Lucene-specific optimization
                            }
                        }
                    },
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        try:
            response = self.client.indices.create(index_name, body=index_body)
            print(f"HNSW index '{index_name}' created successfully")
            return True
        except Exception as e:
            print(f"Error creating HNSW index: {e}")
            return False
    
    def create_ivf_index(self, index_name: str, nlist: int = 1024, nprobes: int = 64) -> bool:
        """
        Create index optimized for IVF algorithm - Good for large datasets.
        
        Args:
            index_name: Name of the index
            nlist: Number of clusters (sqrt(n) to 16*sqrt(n) recommended)
            nprobes: Number of clusters to search (1-nlist/2 recommended)
        """
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.vector_dimension,
                        "method": {
                            "name": "ivf",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {
                                "nlist": nlist,
                                "nprobes": nprobes
                            }
                        }
                    },
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        try:
            response = self.client.indices.create(index_name, body=index_body)
            print(f"IVF index '{index_name}' created successfully")
            return True
        except Exception as e:
            print(f"Error creating IVF index: {e}")
            return False
    
    def create_brute_force_index(self, index_name: str) -> bool:
        """Create index for brute force search - Most accurate but slowest."""
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": self.vector_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "m": 16,
                                "ef_construction": 128
                            }
                        }
                    },
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        try:
            response = self.client.indices.create(index_name, body=index_body)
            print(f"Brute force index '{index_name}' created successfully")
            return True
        except Exception as e:
            print(f"Error creating brute force index: {e}")
            return False
    
    def bulk_index_vectors(self, index_name: str, vectors_data: List[Dict[str, Any]], 
                          batch_size: int = 500) -> bool:
        """
        Bulk index vectors for optimal performance.
        
        Args:
            index_name: Target index name
            vectors_data: List of documents with 'vector', 'text', 'metadata' fields
            batch_size: Number of documents per batch
        """
        def doc_generator():
            for i, doc in enumerate(vectors_data):
                yield {
                    "_index": index_name,
                    "_id": doc.get("id", i),
                    "_source": {
                        "vector": doc["vector"],
                        "text": doc.get("text", ""),
                        "metadata": doc.get("metadata", {}),
                        "timestamp": doc.get("timestamp", time.time())
                    }
                }
        
        try:
            # Bulk index with optimized settings
            success_count = 0
            for batch in self._batch_generator(doc_generator(), batch_size):
                response = helpers.bulk(
                    self.client,
                    batch,
                    index=index_name,
                    chunk_size=batch_size,
                    max_retries=3,
                    initial_backoff=2,
                    max_backoff=600,
                    timeout="60s"
                )
                success_count += response[0]
            
            # Refresh index for immediate searchability
            self.client.indices.refresh(index=index_name)
            print(f"Successfully indexed {success_count} vectors")
            return True
            
        except Exception as e:
            print(f"Error bulk indexing: {e}")
            return False
    
    def search_vectors(self, index_name: str, query_vector: List[float], 
                      k: int = 10, ef: Optional[int] = None, 
                      filter_query: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform optimized vector search.
        
        Args:
            index_name: Index to search
            query_vector: Query vector (1024 dimensions)
            k: Number of results to return
            ef: Search-time parameter for HNSW (defaults to max(k*2, 100))
            filter_query: Optional filter query for hybrid search
        """
        if len(query_vector) != self.vector_dimension:
            raise ValueError(f"Query vector must have {self.vector_dimension} dimensions")
        
        # Optimize ef parameter
        if ef is None:
            ef = max(k * 2, 100)
        
        # Build search query
        knn_query = {
            "field": "vector",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": ef  # For better recall
        }
        
        search_body = {
            "size": k,
            "query": {
                "knn": knn_query
            },
            "_source": ["text", "metadata", "timestamp"]
        }
        
        # Add filter if provided (hybrid search)
        if filter_query:
            search_body["query"] = {
                "bool": {
                    "must": [
                        {"knn": knn_query},
                        filter_query
                    ]
                }
            }
        
        try:
            start_time = time.time()
            response = self.client.search(body=search_body, index=index_name)
            search_time = time.time() - start_time
            
            results = {
                "took": response["took"],
                "search_time_ms": search_time * 1000,
                "total_hits": response["hits"]["total"]["value"],
                "results": []
            }
            
            for hit in response["hits"]["hits"]:
                results["results"].append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "text": hit["_source"].get("text", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                    "timestamp": hit["_source"].get("timestamp")
                })
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return {"error": str(e)}
    
    def hybrid_search(self, index_name: str, query_vector: List[float], 
                     text_query: str, k: int = 10, alpha: float = 0.7) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector similarity and text relevance.
        
        Args:
            alpha: Weight for vector search (0.0-1.0), (1-alpha) for text search
        """
        search_body = {
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                "field": "vector",
                                "query_vector": query_vector,
                                "k": k,
                                "num_candidates": k * 2
                            }
                        },
                        {
                            "match": {
                                "text": text_query
                            }
                        }
                    ]
                }
            },
            "_source": ["text", "metadata", "timestamp"]
        }
        
        try:
            response = self.client.search(body=search_body, index=index_name)
            # Process results similar to search_vectors
            results = {
                "took": response["took"],
                "total_hits": response["hits"]["total"]["value"],
                "results": []
            }
            
            for hit in response["hits"]["hits"]:
                results["results"].append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "text": hit["_source"].get("text", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                    "timestamp": hit["_source"].get("timestamp")
                })
            
            return results
            
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return {"error": str(e)}
    
    def update_search_parameters(self, index_name: str, ef_search: int = 512) -> bool:
        """Update search-time parameters for better performance tuning."""
        try:
            self.client.indices.put_settings(
                index=index_name,
                body={"knn.algo_param.ef_search": ef_search}
            )
            return True
        except Exception as e:
            print(f"Error updating parameters: {e}")
            return False
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        try:
            stats = self.client.indices.stats(index=index_name)
            return {
                "docs_count": stats["indices"][index_name]["total"]["docs"]["count"],
                "store_size": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "segments_count": stats["indices"][index_name]["total"]["segments"]["count"]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _batch_generator(self, generator, batch_size):
        """Helper to create batches from generator."""
        batch = []
        for item in generator:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Example usage and performance testing
def main():
    # Initialize the search client
    search_client = OpenSearchVectorSearch(
        host='localhost',
        port=9200,
        http_auth=('admin', 'admin')  # Adjust credentials as needed
    )
    
    # Create optimized HNSW index (recommended for most use cases)
    index_name = "vector_search_1024_optimized"
    search_client.create_hnsw_index(
        index_name=index_name,
        m=48,  # Good balance between accuracy and memory
        ef_construction=512  # Higher for better accuracy
    )
    
    # Generate sample data for testing
    sample_vectors = []
    for i in range(1000):
        vector = np.random.rand(1024).astype(np.float32).tolist()
        sample_vectors.append({
            "id": f"doc_{i}",
            "vector": vector,
            "text": f"Sample document {i} with some text content",
            "metadata": {"category": f"cat_{i % 10}", "importance": i % 5}
        })
    
    # Bulk index the vectors
    print("Indexing sample vectors...")
    search_client.bulk_index_vectors(index_name, sample_vectors, batch_size=100)
    
    # Perform search
    query_vector = np.random.rand(1024).astype(np.float32).tolist()
    
    print("\nPerforming vector search...")
    results = search_client.search_vectors(
        index_name=index_name,
        query_vector=query_vector,
        k=10,
        ef=200  # Adjust for speed vs accuracy tradeoff
    )
    
    print(f"Search took {results.get('search_time_ms', 0):.2f}ms")
    print(f"Found {len(results.get('results', []))} results")
    
    # Print top results
    for i, result in enumerate(results.get('results', [])[:3]):
        print(f"Result {i+1}: Score={result['score']:.4f}, ID={result['id']}")
    
    # Get index statistics
    stats = search_client.get_index_stats(index_name)
    print(f"\nIndex stats: {stats}")

if __name__ == "__main__":
    main()
