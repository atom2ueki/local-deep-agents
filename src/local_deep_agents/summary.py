from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from local_deep_agents.utils import get_today_str
from local_deep_agents.states import Summary
from local_deep_agents.prompts import SUMMARIZE_WEB_SEARCH

# Summarization model
summarization_model = init_chat_model(model="openai:gpt-4o-mini")

def summarize_webpage_content(webpage_content: str) -> Summary:
  """Summarize webpage content using the configured summarization model.

  Args:
      webpage_content: Raw webpage content to summarize

  Returns:
      Summary object with filename and summary
  """
  try:
      # Set up structured output model for summarization
      structured_model = summarization_model.with_structured_output(Summary)

      # Generate summary
      summary_and_filename = structured_model.invoke([
          HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
              webpage_content=webpage_content,
              date=get_today_str()
          ))
      ])

      return summary_and_filename

  except Exception:
      # Return a basic summary object on failure
      return Summary(
          filename="search_result.md",
          summary=webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
      )
