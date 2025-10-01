import os
import uuid, base64

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from typing_extensions import Annotated, Literal

from local_deep_agents.utils import get_today_str
from local_deep_agents.tavily import run_tavily_search
from local_deep_agents.states import DeepAgentState, Summary
from local_deep_agents.summary import summarize_webpage_content

def process_search_results(results: dict) -> list[dict]:
  """Process search results by summarizing content where available.

  Args:
      results: Tavily search results dictionary

  Returns:
      List of processed results with summaries
  """
  processed_results = []

  # Create a client for HTTP requests
  HTTPX_CLIENT = httpx.Client()

  for result in results.get('results', []):

      # Get url
      url = result['url']

      # Read url
      response = HTTPX_CLIENT.get(url)

      if response.status_code == 200:
          # Convert HTML to markdown
          raw_content = markdownify(response.text)
          summary_obj = summarize_webpage_content(raw_content)
      else:
          # Use Tavily's generated summary
          raw_content = result.get('raw_content', '')
          summary_obj = Summary(
              filename="URL_error.md",
              summary=result.get('content', 'Error reading URL; try another search.')
          )

      # uniquify file names
      uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
      name, ext = os.path.splitext(summary_obj.filename)
      summary_obj.filename = f"{name}_{uid}{ext}"

      processed_results.append({
          'url': result['url'],
          'title': result['title'],
          'summary': summary_obj.summary,
          'filename': summary_obj.filename,
          'raw_content': raw_content,
      })

  return processed_results

@tool(parse_docstring=True)
def tavily_search(
  query: str,
  state: Annotated[DeepAgentState, InjectedState],
  tool_call_id: Annotated[str, InjectedToolCallId],
  max_results: Annotated[int, InjectedToolArg] = 1,
  topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
  """Search web and save detailed results to files while returning minimal context.

  Performs web search and saves full content to files for context offloading.
  Returns only essential information to help the agent decide on next steps.

  Args:
      query: Search query to execute
      state: Injected agent state for file storage
      tool_call_id: Injected tool call identifier
      max_results: Maximum number of results to return (default: 1)
      topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

  Returns:
      Command that saves full results to files and provides minimal summary
  """
  # Execute search
  search_results = run_tavily_search(
      query,
      max_results=max_results,
      topic=topic,
      include_raw_content=True,
  )

  # Process and summarize results
  processed_results = process_search_results(search_results)

  # Save each result to a file and prepare summary
  files = state.get("files", {})
  saved_files = []
  summaries = []

  for i, result in enumerate(processed_results):
      # Use the AI-generated filename from summarization
      filename = result['filename']

      # Create file content with full details
      file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""

      files[filename] = file_content
      saved_files.append(filename)
      summaries.append(f"- {filename}: {result['summary']}...")

  # Create minimal summary for tool message - focus on what was collected
  summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""

  return Command(
      update={
          "files": files,
          "messages": [
              ToolMessage(summary_text, tool_call_id=tool_call_id)
          ],
      }
  )
