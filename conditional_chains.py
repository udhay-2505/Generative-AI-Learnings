from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

# Load environment variables
load_dotenv()

# Initialize a single Groq model
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3  # lower temperature for classification tasks
)

# Output parsers
parser_str = StrOutputParser()

# Pydantic schema for structured output
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser_pydantic = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for sentiment classification
prompt_classify = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text as either positive or negative.\n\n"
        "Feedback: {feedback}\n\n"
        "{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser_pydantic.get_format_instructions()},
)

# Chain for classification
classifier_chain = prompt_classify | model | parser_pydantic

# Prompts for generating responses
prompt_positive = PromptTemplate(
    template="Write an appropriate and cheerful response to this positive feedback:\n\n{feedback}",
    input_variables=["feedback"],
)

prompt_negative = PromptTemplate(
    template="Write an empathetic and constructive response to this negative feedback:\n\n{feedback}",
    input_variables=["feedback"],
)

# Conditional branching chain
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | model | parser_str),
    (lambda x: x.sentiment == "negative", prompt_negative | model | parser_str),
    RunnableLambda(lambda x: "Could not determine sentiment confidently.")
)

# Final pipeline
chain = classifier_chain | branch_chain

# Test run
feedback_text = "Oh great, Exactly what I needed!"
response = chain.invoke({"feedback": feedback_text})

print("🧠 Feedback:", feedback_text)
print("💬 Response:", response)

# Optional: visualize the logic graph
chain.get_graph().print_ascii()
