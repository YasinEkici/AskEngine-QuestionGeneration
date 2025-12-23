from pydantic import BaseModel, Field

class ContextAnswerPair(BaseModel):
    context: str = Field(..., description="The paragraph or text context.")
    answer: str = Field(..., description="The answer for which to generate a question.")
    beam_size: int = Field(default=4, description="Number of beams for beam search.")
    max_length: int = Field(default=64, description="Maximum length of the generated question.")
    num_candidates: int = Field(default=1, description="Number of candidate questions to return.")

class GeneratedQuestion(BaseModel):
    question: str = Field(..., description="The best generated question.")
    candidates: list[str] = Field(default_factory=list, description="List of all candidate questions (reranked).")
    model_name: str = Field(..., description="Name of the model used.")
    meta: dict = Field(default_factory=dict, description="Additional metadata related to the generation process.")

class SuggestionRequest(BaseModel):
    context: str
    max_suggestions: int = 5

class SuggestionResponse(BaseModel):
    candidates: list[str]
