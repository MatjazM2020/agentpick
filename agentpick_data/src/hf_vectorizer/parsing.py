"""README parsing and text chunking utilities."""

import logging
from typing import List, Dict
from pathlib import Path

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ReadmeParser:
    """Parser for README files and text chunking."""
    
    def __init__(self, tokenizer_model: str = "BAAI/bge-large-en-v1.5"):
        """Initialize parser with tokenizer.
        
        Args:
            tokenizer_model: HuggingFace model ID for tokenizer
        """
        logger.info(f"Loading tokenizer for {tokenizer_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        logger.info("Tokenizer loaded")
    
    def parse_readme(self, readme_path: Path) -> List[Dict[str, str]]:
        """Parse README into sections by ## headers.
        
        Args:
            readme_path: Path to README file
            
        Returns:
            List of sections with 'header' and 'content'
        """
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Error reading README: {e}")
            return []
        
        sections = []
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        in_frontmatter = False
        
        for line in lines:
            # Skip YAML frontmatter
            if line.strip() == '---':
                in_frontmatter = not in_frontmatter
                continue
            
            if in_frontmatter:
                continue
            
            # Check for section header
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        sections.append({
                            'header': current_section,
                            'content': content_text
                        })
                
                # Start new section
                current_section = line[3:].strip()
                current_content = []
            
            elif current_section is not None:
                current_content.append(line)
        
        # Add last section
        if current_section:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                sections.append({
                    'header': current_section,
                    'content': content_text
                })
        
        # If no sections found, use entire content as single section
        if not sections and content.strip():
            sections.append({
                'header': 'Full Content',
                'content': content.strip()
            })
        
        return sections
    
    def chunk_text_tokens(
        self,
        text: str,
        max_tokens: int = 512,
        overlap: int = 64
    ) -> List[str]:
        """Chunk text into overlapping token-based chunks.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start = 0
        n = len(tokens)
        
        while start < n:
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            if end >= n:
                break
            
            start += max_tokens - overlap
        
        return chunks
