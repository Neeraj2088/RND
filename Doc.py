import re
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    content: str
    start_idx: int
    end_idx: int
    chunk_id: str
    metadata: Dict = None
    embedding: Optional[List[float]] = None

class HTMLCleaner:
    """Clean and preprocess HTML content"""
    
    def __init__(self):
        self.unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']
        
    def clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract meaningful text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted tags
        for tag in soup(self.unwanted_tags):
            tag.decompose()
        
        # Extract text while preserving some structure
        text = self._extract_structured_text(soup)
        
        # Clean up whitespace and normalize
        text = self._normalize_text(text)
        
        return text
    
    def _extract_structured_text(self, soup) -> str:
        """Extract text while preserving document structure"""
        text_parts = []
        
        # Handle headings specially to preserve hierarchy
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(heading.name[1])
            text_parts.append(f"\n{'#' * level} {heading.get_text().strip()}\n")
        
        # Handle paragraphs
        for p in soup.find_all('p'):
            if p.get_text().strip():
                text_parts.append(p.get_text().strip() + "\n")
        
        # Handle lists
        for ul in soup.find_all(['ul', 'ol']):
            for li in ul.find_all('li'):
                text_parts.append(f"• {li.get_text().strip()}\n")
        
        # Handle tables
        for table in soup.find_all('table'):
            text_parts.append(self._extract_table_text(table))
        
        # Fallback: get remaining text
        remaining_text = soup.get_text()
        if remaining_text.strip():
            text_parts.append(remaining_text)
        
        return " ".join(text_parts)
    
    def _extract_table_text(self, table) -> str:
        """Extract text from tables in a structured way"""
        rows = []
        for tr in table.find_all('tr'):
            cells = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
            if any(cells):  # Only add non-empty rows
                rows.append(" | ".join(cells))
        return "\n".join(rows) + "\n"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by cleaning whitespace and formatting"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Clean up special characters
        text = re.sub(r'[^\w\s\n\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\#\•]', '', text)
        return text.strip()

class EmbeddingClient:
    """Client for local embedding API"""
    
    def __init__(self, api_url: str, timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from local API"""
        try:
            response = requests.post(
                self.api_url,
                json={"text": text},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            response = requests.post(
                self.api_url + "/batch",  # Assuming batch endpoint
                json={"texts": texts},
                timeout=self.timeout * 2
            )
            response.raise_for_status()
            return response.json().get("embeddings", [])
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}, falling back to individual requests")
            return [self.get_embedding(text) for text in texts]

class DocumentChunker:
    """Advanced document chunking strategies for RAG"""
    
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def fixed_size_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Chunk]:
        """Simple fixed-size chunking with overlap"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text,
                    start_idx=i,
                    end_idx=min(i + chunk_size, len(words)),
                    chunk_id=f"chunk_{len(chunks)}",
                    metadata={"method": "fixed_size", "word_count": len(chunk_words)}
                ))
        
        return chunks
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7, max_chunk_size: int = 1500) -> List[Chunk]:
        """Semantic chunking based on sentence similarity"""
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(content=text, start_idx=0, end_idx=len(text), chunk_id="chunk_0")]
        
        # Get embeddings for sentences
        embeddings = self.embedding_client.get_embeddings_batch(sentences)
        if not embeddings or len(embeddings) != len(sentences):
            logger.warning("Failed to get embeddings, falling back to fixed chunking")
            return self.fixed_size_chunking(text)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_start = 0
        
        for i in range(1, len(sentences)):
            # Calculate similarity between current sentence and chunk
            chunk_embedding = self._average_embeddings([embeddings[j] for j in range(current_start, i)])
            sentence_embedding = embeddings[i]
            
            similarity = self._cosine_similarity(chunk_embedding, sentence_embedding)
            chunk_text = " ".join(current_chunk + [sentences[i]])
            
            # Check if we should continue current chunk or start new one
            if (similarity >= similarity_threshold and 
                self.count_tokens(chunk_text) <= max_chunk_size):
                current_chunk.append(sentences[i])
            else:
                # Finalize current chunk
                chunk_content = " ".join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_content,
                    start_idx=current_start,
                    end_idx=i,
                    chunk_id=f"semantic_chunk_{len(chunks)}",
                    metadata={"method": "semantic", "similarity_threshold": similarity_threshold}
                ))
                
                # Start new chunk
                current_chunk = [sentences[i]]
                current_start = i
        
        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_content,
                start_idx=current_start,
                end_idx=len(sentences),
                chunk_id=f"semantic_chunk_{len(chunks)}",
                metadata={"method": "semantic", "similarity_threshold": similarity_threshold}
            ))
        
        return chunks
    
    def hierarchical_chunking(self, text: str, chunk_sizes: List[int] = [500, 1000, 2000]) -> Dict[str, List[Chunk]]:
        """Create hierarchical chunks at different granularities"""
        hierarchical_chunks = {}
        
        for size in chunk_sizes:
            chunks = self.fixed_size_chunking(text, chunk_size=size, overlap=size//10)
            hierarchical_chunks[f"level_{size}"] = chunks
        
        return hierarchical_chunks
    
    def structure_aware_chunking(self, text: str, max_chunk_size: int = 1200) -> List[Chunk]:
        """Chunk based on document structure (headings, paragraphs)"""
        chunks = []
        sections = self._split_by_structure(text)
        
        current_chunk = ""
        current_metadata = {}
        
        for section in sections:
            section_text = section["content"]
            section_type = section["type"]
            
            # If adding this section would exceed max size, finalize current chunk
            if (current_chunk and 
                self.count_tokens(current_chunk + " " + section_text) > max_chunk_size):
                
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    start_idx=len(chunks),
                    end_idx=len(chunks) + 1,
                    chunk_id=f"struct_chunk_{len(chunks)}",
                    metadata={**current_metadata, "method": "structure_aware"}
                ))
                current_chunk = ""
                current_metadata = {}
            
            # Add section to current chunk
            if current_chunk:
                current_chunk += " " + section_text
            else:
                current_chunk = section_text
                current_metadata = {"primary_type": section_type}
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                start_idx=len(chunks),
                end_idx=len(chunks) + 1,
                chunk_id=f"struct_chunk_{len(chunks)}",
                metadata={**current_metadata, "method": "structure_aware"}
            ))
        
        return chunks
    
    def adaptive_chunking(self, text: str, target_chunk_size: int = 1000, 
                         similarity_threshold: float = 0.65) -> List[Chunk]:
        """Adaptive chunking that combines multiple strategies"""
        # First, try structure-aware chunking
        struct_chunks = self.structure_aware_chunking(text, target_chunk_size)
        
        # If chunks are too large, apply semantic chunking
        final_chunks = []
        for chunk in struct_chunks:
            if self.count_tokens(chunk.content) > target_chunk_size * 1.5:
                # Apply semantic chunking to large chunks
                semantic_chunks = self.semantic_chunking(
                    chunk.content, 
                    similarity_threshold, 
                    target_chunk_size
                )
                final_chunks.extend(semantic_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with spacy or nltk
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_structure(self, text: str) -> List[Dict]:
        """Split text by structural elements"""
        sections = []
        lines = text.split('\n')
        
        current_section = ""
        current_type = "paragraph"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a heading
            if line.startswith('#'):
                if current_section:
                    sections.append({"content": current_section, "type": current_type})
                current_section = line
                current_type = "heading"
            elif line.startswith('•'):
                if current_type != "list":
                    if current_section:
                        sections.append({"content": current_section, "type": current_type})
                    current_section = line
                    current_type = "list"
                else:
                    current_section += " " + line
            else:
                if current_type != "paragraph":
                    if current_section:
                        sections.append({"content": current_section, "type": current_type})
                    current_section = line
                    current_type = "paragraph"
                else:
                    current_section += " " + line
        
        if current_section:
            sections.append({"content": current_section, "type": current_type})
        
        return sections
    
    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate average of embeddings"""
        if not embeddings:
            return []
        return np.mean(embeddings, axis=0).tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class RAGDocumentProcessor:
    """Main class that orchestrates the entire RAG preprocessing pipeline"""
    
    def __init__(self, embedding_api_url: str):
        self.html_cleaner = HTMLCleaner()
        self.embedding_client = EmbeddingClient(embedding_api_url)
        self.chunker = DocumentChunker(self.embedding_client)
    
    def process_html_content(self, html_content: str, 
                           chunking_strategy: str = "adaptive",
                           **chunking_params) -> List[Chunk]:
        """Process HTML content end-to-end"""
        
        logger.info("Cleaning HTML content...")
        clean_text = self.html_cleaner.clean_html(html_content)
        
        logger.info(f"Applying {chunking_strategy} chunking strategy...")
        chunks = self._apply_chunking_strategy(clean_text, chunking_strategy, **chunking_params)
        
        logger.info("Generating embeddings for chunks...")
        chunks_with_embeddings = self._add_embeddings_to_chunks(chunks)
        
        logger.info(f"Processing complete. Generated {len(chunks_with_embeddings)} chunks.")
        return chunks_with_embeddings
    
    def _apply_chunking_strategy(self, text: str, strategy: str, **params) -> List[Chunk]:
        """Apply the specified chunking strategy"""
        if strategy == "fixed":
            return self.chunker.fixed_size_chunking(text, **params)
        elif strategy == "semantic":
            return self.chunker.semantic_chunking(text, **params)
        elif strategy == "hierarchical":
            hierarchical = self.chunker.hierarchical_chunking(text, **params)
            return hierarchical.get("level_1000", [])  # Return default level
        elif strategy == "structure":
            return self.chunker.structure_aware_chunking(text, **params)
        elif strategy == "adaptive":
            return self.chunker.adaptive_chunking(text, **params)
        else:
            logger.warning(f"Unknown strategy: {strategy}, using adaptive")
            return self.chunker.adaptive_chunking(text, **params)
    
    def _add_embeddings_to_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add embeddings to chunks"""
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_client.get_embeddings_batch(chunk_texts)
        
        for i, chunk in enumerate(chunks):
            if i < len(embeddings) and embeddings[i]:
                chunk.embedding = embeddings[i]
        
        return chunks

# Example usage
def main():
    # Initialize the processor
    processor = RAGDocumentProcessor("http://localhost:8000/embed")  # Your local embedding API
    
    # Example HTML content
    html_content = """
    <html>
    <head><title>Sample Document</title></head>
    <body>
        <h1>Introduction to Machine Learning</h1>
        <p>Machine learning is a subset of artificial intelligence...</p>
        <h2>Supervised Learning</h2>
        <p>Supervised learning involves training models on labeled data...</p>
        <ul>
            <li>Classification problems</li>
            <li>Regression problems</li>
        </ul>
        <h2>Unsupervised Learning</h2>
        <p>Unsupervised learning works with unlabeled data...</p>
    </body>
    </html>
    """
    
    # Process with different strategies
    strategies = [
        ("adaptive", {}),
        ("semantic", {"similarity_threshold": 0.7}),
        ("structure", {"max_chunk_size": 1200}),
        ("fixed", {"chunk_size": 800, "overlap": 100})
    ]
    
    for strategy, params in strategies:
        print(f"\n=== Testing {strategy.upper()} chunking ===")
        chunks = processor.process_html_content(html_content, strategy, **params)
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} ({len(chunk.content)} chars):")
            print(f"Content: {chunk.content[:100]}...")
            print(f"Metadata: {chunk.metadata}")
            print(f"Has embedding: {chunk.embedding is not None}")
            print("-" * 50)

if __name__ == "__main__":
    main()
