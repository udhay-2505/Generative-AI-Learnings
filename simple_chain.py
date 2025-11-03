from langchain_groq import ChatGroq
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq model
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.7  
)

# Define a simple prompt template
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}.",
    input_variables=["topic"]
)

# Output parser converts model output to string
parser = StrOutputParser()  

# Create the runnable chain
chain = prompt | model | parser  

# Run the chain
result = chain.invoke({"topic": "India"})
print(result)

# chain visualize
        #chain.get_graph().print_ascii()
