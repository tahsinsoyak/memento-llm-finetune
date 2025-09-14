# tools/document_tool.py
import requests
import os
from urllib.parse import urlparse
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("Warning: markitdown not available. Install with: pip install markitdown")
from bs4 import BeautifulSoup

class DocumentTool:
    """Fetches and parses content from URLs or local files using markitdown."""
    
    def __init__(self, max_content_length: int = 50000):
        """
        Initialize document tool.
        
        :param max_content_length: Maximum length of content to return
        """
        self.max_content_length = max_content_length
        self.markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None
    
    def fetch_content(self, source: str) -> str:
        """
        Fetch and process content from URL or file path.
        
        :param source: URL or file path to fetch content from
        :return: Processed content as text
        """
        try:
            # Check if it's a URL or file path
            parsed = urlparse(source)
            if parsed.scheme in ('http', 'https'):
                return self._fetch_url_content(source)
            elif os.path.exists(source):
                return self._fetch_file_content(source)
            else:
                return f"Error: Source '{source}' is not a valid URL or file path"
        except Exception as e:
            return f"Error processing source '{source}': {str(e)}"
    
    def _fetch_url_content(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            
            # Get content type
            content_type = resp.headers.get('Content-Type', '').lower()
            
            # Use markitdown to convert content
            if 'html' in content_type:
                # For HTML, use BeautifulSoup first to clean up
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                cleaned_html = str(soup)
                
                # Try to use markitdown via temporary file approach
                if self.markitdown:
                    try:
                        result = self.markitdown.convert(cleaned_html)
                        content = result.text_content if hasattr(result, 'text_content') else str(result)
                    except Exception:
                        # Fallback to simple text extraction
                        content = soup.get_text(separator=' ', strip=True)
                else:
                    # Fallback to simple text extraction
                    content = soup.get_text(separator=' ', strip=True)
            else:
                # For other content types, try markitdown directly
                if self.markitdown:
                    try:
                        result = self.markitdown.convert(resp.text)
                        content = result.text_content if hasattr(result, 'text_content') else str(result)
                    except Exception:
                        content = resp.text
                else:
                    content = resp.text
            
            # Clean up and truncate if necessary
            content = self._clean_and_truncate(content)
            return f"Content from {url}:\n\n{content}"
            
        except requests.RequestException as e:
            return f"Error fetching URL {url}: {str(e)}"
        except Exception as e:
            return f"Error processing URL content: {str(e)}"
    
    def _fetch_file_content(self, file_path: str) -> str:
        """Fetch content from a local file."""
        try:
            # Use markitdown to convert file if available
            if self.markitdown:
                result = self.markitdown.convert(file_path)
                content = result.text_content if hasattr(result, 'text_content') else str(result)
            else:
                # Fallback to simple file reading
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Clean up and truncate if necessary
            content = self._clean_and_truncate(content)
            return f"Content from {file_path}:\n\n{content}"
            
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
    
    def _clean_and_truncate(self, content: str) -> str:
        """Clean and truncate content if necessary."""
        # Remove excessive whitespace
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only keep non-empty lines
                cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n\n[Content truncated for length...]"
        
        return content
    
    def get_page_summary(self, url: str, summary_length: int = 500) -> str:
        """
        Get a brief summary of a web page.
        
        :param url: URL to summarize
        :param summary_length: Maximum length of summary
        :return: Brief summary of the page content
        """
        content = self._fetch_url_content(url)
        if content.startswith("Error"):
            return content
        
        # Extract just the first part for summary
        content_start = content.split('\n\n', 1)[-1]  # Remove the "Content from..." header
        if len(content_start) > summary_length:
            # Find a good breaking point (end of sentence)
            truncated = content_start[:summary_length]
            last_period = truncated.rfind('.')
            if last_period > summary_length // 2:
                truncated = truncated[:last_period + 1]
            content_start = truncated + "..."
        
        return f"Summary of {url}:\n{content_start}"
