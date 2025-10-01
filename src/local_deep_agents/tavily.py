from tavily import TavilyClient
from typing_extensions import Literal

tavily_client = TavilyClient()

def run_tavily_search(
  search_query: str,
  max_results: int = 1,
  topic: Literal["general", "news", "finance"] = "general",
  include_raw_content: bool = True,
) -> dict:
  """Perform search using Tavily API for a single query.

  Args:
      search_query: Search query to execute
      max_results: Maximum number of results per query
      topic: Topic filter for search results
      include_raw_content: Whether to include raw webpage content

  Returns:
      Search results dictionary
  """
  result = tavily_client.search(
      search_query,
      max_results=max_results,
      include_raw_content=include_raw_content,
      topic=topic
  )

  return result
