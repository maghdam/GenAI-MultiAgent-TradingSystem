import requests
from bs4 import BeautifulSoup
import logging
from .llm_analyzer import _ollama_generate, MODEL_DEFAULT

# Set a user agent to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def _fetch_search_results(query: str, num_results: int = 3) -> list[str]:
    """Fetches the top search result URLs from DuckDuckGo."""
    search_url = f"https://html.duckduckgo.com/html/?q={query}"
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', class_='result__a')]
        return links[:num_results]
    except Exception as e:
        logging.error(f"Web search failed for query '{query}': {e}")
        return []

def _scrape_article_text(url: str) -> str:
    """Scrapes the main text content from a given URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # A simple heuristic: join all paragraph texts
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except Exception as e:
        logging.warning(f"Failed to scrape {url}: {e}")
        return ""

# List of trusted financial news sources
TRUSTED_SOURCES = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "investing.com",
    "marketwatch.com",
    "forexlive.com"
]

def get_news_summary(topic: str) -> str:
    """Searches for news on a topic across trusted sites, scrapes the top results, and returns an LLM-generated summary."""
    logging.info(f"Fetching news summary for topic: {topic} from trusted sources.")

    # Create a list of search queries for DuckDuckGo
    search_queries = [f"financial news {topic}"] # General search first
    for source in TRUSTED_SOURCES:
        search_queries.append(f"site:{source} {topic} news")

    all_urls = []
    for query in search_queries:
        # Fetch top 2 for general, top 1 for specific sites
        num_results = 2 if "site:" not in query else 1
        urls = _fetch_search_results(query, num_results=num_results)
        if urls:
            all_urls.extend(urls)
    
    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(all_urls))

    if not unique_urls:
        return f"I couldn't find any recent news for '{topic}' from trusted sources."

    # Scrape content from the top 5 unique URLs to keep it efficient
    content = []
    for url in unique_urls[:5]:
        scraped_text = _scrape_article_text(url)
        if scraped_text:
            content.append(scraped_text)
    
    if not content:
        return f"I found some articles for '{topic}', but I was unable to read their content."

    combined_text = '\n\n---\n\n'.join(content)

    prompt = f"""You are a financial analyst. Below is the text from several recent news articles on the topic of '{topic}'.

Read all the articles and provide a concise, one-paragraph summary of the key information and market sentiment. Focus on the most important takeaways.

Article Texts:
{combined_text}

Summary:"""

    try:
        summary = _ollama_generate(
            prompt=prompt,
            model=MODEL_DEFAULT,
            timeout=60, # Longer timeout for summarization
            json_only=False,
            options_overrides={"num_predict": 256}
        )
        return summary
    except Exception as e:
        logging.error(f"Failed to generate news summary for '{topic}': {e}")
        return "I found some news, but I had trouble summarizing it."
