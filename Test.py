import re
from typing import List, Optional

class TextChunker:
    """Basic text chunking strategies for various use cases."""
    
    @staticmethod
    def chunk_by_character_count(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into chunks by character count with optional overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
            
        return chunks
    
    @staticmethod
    def chunk_by_sentences(text: str, max_sentences: int = 5) -> List[str]:
        """
        Split text into chunks by sentence count.
        
        Args:
            text: Input text to chunk
            max_sentences: Maximum sentences per chunk
            
        Returns:
            List of text chunks
        """
        # Simple sentence splitting (you might want to use nltk or spaCy for better results)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i + max_sentences]
            chunk = '. '.join(chunk_sentences)
            if chunk:
                chunks.append(chunk + '.')
                
        return chunks
    
    @staticmethod
    def chunk_by_paragraphs(text: str, max_paragraphs: int = 3) -> List[str]:
        """
        Split text into chunks by paragraph count.
        
        Args:
            text: Input text to chunk
            max_paragraphs: Maximum paragraphs per chunk
            
        Returns:
            List of text chunks
        """
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        for i in range(0, len(paragraphs), max_paragraphs):
            chunk_paragraphs = paragraphs[i:i + max_paragraphs]
            chunk = '\n\n'.join(chunk_paragraphs)
            chunks.append(chunk)
            
        return chunks
    
    @staticmethod
    def chunk_by_words(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
        """
        Split text into chunks by word count with optional overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        
        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")
            
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = end - overlap
            
        return chunks
    
    @staticmethod
    def smart_chunk(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Smart chunking that tries to split at sentence boundaries when possible.
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        if max_chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a sentence boundary near the end
            chunk_text = text[start:end]
            
            # Look for sentence endings in the last 200 characters
            search_start = max(0, len(chunk_text) - 200)
            last_part = chunk_text[search_start:]
            
            # Find the last sentence ending
            sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', last_part)]
            
            if sentence_endings:
                # Adjust end to the last sentence boundary
                last_sentence_end = sentence_endings[-1]
                actual_end = start + search_start + last_sentence_end
                chunk = text[start:actual_end]
            else:
                # No sentence boundary found, use character limit
                chunk = text[start:end]
                actual_end = end
            
            chunks.append(chunk.strip())
            start = actual_end - overlap
            
        return chunks

# Example usage
if __name__ == "__main__":
    sample_text = """
    This is the first paragraph. It contains multiple sentences. Each sentence provides some information.
    
    This is the second paragraph. It also has several sentences. The content continues here with more details.
    
    Here's a third paragraph. It demonstrates how the chunking algorithms work. You can see different strategies in action.
    
    Finally, this is the last paragraph. It wraps up our example text. The chunking methods will process all of this content.
    """
    
    chunker = TextChunker()
    
    print("=== Character-based chunking ===")
    char_chunks = chunker.chunk_by_character_count(sample_text, chunk_size=200, overlap=50)
    for i, chunk in enumerate(char_chunks):
        print(f"Chunk {i+1}: {chunk[:100]}...")
    
    print("\n=== Sentence-based chunking ===")
    sentence_chunks = chunker.chunk_by_sentences(sample_text, max_sentences=2)
    for i, chunk in enumerate(sentence_chunks):
        print(f"Chunk {i+1}: {chunk}")
    
    print("\n=== Paragraph-based chunking ===")
    para_chunks = chunker.chunk_by_paragraphs(sample_text, max_paragraphs=2)
    for i, chunk in enumerate(para_chunks):
        print(f"Chunk {i+1}: {chunk[:100]}...")
    
    print("\n=== Word-based chunking ===")
    word_chunks = chunker.chunk_by_words(sample_text, chunk_size=30, overlap=5)
    for i, chunk in enumerate(word_chunks):
        print(f"Chunk {i+1}: {chunk}")
    
    print("\n=== Smart chunking ===")
    smart_chunks = chunker.smart_chunk(sample_text, max_chunk_size=250, overlap=30)
    for i, chunk in enumerate(smart_chunks):
        print(f"Chunk {i+1}: {chunk}")
