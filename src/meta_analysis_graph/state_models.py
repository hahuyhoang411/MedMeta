from typing import List, Annotated
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# --- Pydantic Models for Structured Output ---

class ResearchPlan(BaseModel):
     """Structured research plan."""
     background: str = Field(description="Brief background context for the research topic.")
     key_questions: List[str] = Field(description="List of 5 specific research questions the plan aims to address.")
     search_strategy_summary: str = Field(description="A brief summary of the types of studies or concepts to search for.")

class SearchQueries(BaseModel):
    """Structured list of search queries optimized for literature databases."""
    queries: List[str] = Field(
        description="List of 5 optimized search queries based on the research plan, suitable for PubMed or similar databases (e.g., using keywords)."
    )

# --- LangGraph State Definition ---

class MetaAnalysisState(TypedDict):
    """Represents the state of our meta-analysis assistant graph."""
    research_topic: str           # Input topic from the researcher
    research_plan: ResearchPlan   # Generated plan
    search_queries: List[str]     # Optimized queries for retrieval
    retrieved_docs: Annotated[List[Document], operator.add] # Aggregated documents from all queries
    final_conclusion: str         # Final synthesized conclusion (renamed from meta_analysis_conclusion for clarity)

    # Temporary state for parallel execution (managed by LangGraph)
    current_query: str            # The query currently being processed by a branch
    current_docs: List[Document]  # Docs retrieved for the current_query