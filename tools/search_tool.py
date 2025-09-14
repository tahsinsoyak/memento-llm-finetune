# tools/search_tool.py
from typing import List, Dict, Optional

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
        print("Warning: Using deprecated duckduckgo_search. Consider upgrading to ddgs: pip install ddgs")
    except ImportError:
        DDGS_AVAILABLE = False
        print("Error: Neither ddgs nor duckduckgo_search is available. Install with: pip install ddgs")

class SearchTool:
    """Uses DuckDuckGo free search API to perform web searches."""
    
    def __init__(self, max_results: int = 5):
        """
        Initialize search tool with DuckDuckGo.
        
        :param max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        if not DDGS_AVAILABLE:
            raise ImportError("DDGS not available. Install with: pip install ddgs")
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Perform web search using DuckDuckGo.
        
        :param query: Search query string
        :param max_results: Override default max_results for this search
        :return: Formatted search results as text
        """
        try:
            results_limit = max_results or self.max_results
            results = list(self.ddgs.text(
                keywords=query,
                max_results=results_limit,
                region='wt-wt',
                safesearch='moderate'
            ))
            
            if not results:
                return f"No search results found for: {query}"
            
            # Format results as text
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                href = result.get('href', 'No URL')
                
                formatted_result = f"""
Result {i}:
Title: {title}
Description: {body}
URL: {href}
---"""
                formatted_results.append(formatted_result)
            
            return f"Search results for '{query}':\n" + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing search for '{query}': {str(e)}"
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Perform news search using DuckDuckGo.
        
        :param query: Search query string
        :param max_results: Override default max_results for this search
        :return: Formatted news results as text
        """
        try:
            results_limit = max_results or self.max_results
            results = list(self.ddgs.news(
                keywords=query,
                max_results=results_limit,
                region='wt-wt',
                safesearch='moderate'
            ))
            
            if not results:
                return f"No news results found for: {query}"
            
            # Format results as text
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                url = result.get('url', 'No URL')
                date = result.get('date', 'No date')
                source = result.get('source', 'Unknown source')
                
                formatted_result = f"""
News {i}:
Title: {title}
Source: {source}
Date: {date}
Description: {body}
URL: {url}
---"""
                formatted_results.append(formatted_result)
            
            return f"News results for '{query}':\n" + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing news search for '{query}': {str(e)}"
    
    def get_results_raw(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Get raw search results as list of dictionaries.
        
        :param query: Search query string
        :param max_results: Override default max_results for this search
        :return: List of result dictionaries
        """
        try:
            results_limit = max_results or self.max_results
            results = list(self.ddgs.text(
                keywords=query,
                max_results=results_limit,
                region='wt-wt',
                safesearch='moderate'
            ))
            return results
        except Exception as e:
            print(f"Error getting raw search results: {e}")
            return []
