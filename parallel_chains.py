from langchain_groq import ChatGroq
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel


# Load environment variables from .env file
load_dotenv()

model= ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile", 
)
parser= StrOutputParser()

prompt_notes=PromptTemplate(
    template='generate a short and simple notes from the following text\n {text}'
    , input_variables=['text']
)

prompt_quiz=PromptTemplate(
    template='generate a quiz of 5 short questions and answers from tthe follwing text\n {text}',
    input_variables=['text']
)

prompt_merge=PromptTemplate(
    template='Merge the provided notes and quiz into a single document\n notes-> {notes} ans quiz-> {quiz}',
    input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parallel_chain= RunnableParallel({
    'notes': prompt_notes| model | parser,
    'quiz': prompt_quiz | model | parser
})

merge_chain= prompt_merge | model | parser
chain = parallel_chain | merge_chain

text="""
    Here are **definitions for each of the 6 core LangChain components** you mentioned—**Models, Agents, Memory, Index (Indexes), Chains, Prompts**—based directly on authoritative and clear explanations:

***

**1. Models**
- These are the core interfaces through which you interact with AI. LangChain includes:
    - **Language Models:** AI systems that process, generate, and understand text, e.g., OpenAI's GPT.
    - **Chat Models:** Specialized LLMs for conversations that take a sequence of messages as input.
    - **Embedding Models:** Generate vector embeddings for semantic search and retrieval tasks.
- **Purpose:** They abstract away the complexity of different AI backends, offering a unified interface to interact with various LLMs and embedding models.

***

**2. Agents**
- **Agents** are autonomous, LLM-powered systems that can make decisions and perform tasks using external tools or APIs (for example: search, calculator, database) to accomplish goals.
- **ReAct Pattern:** Allows an LLM to alternate between reasoning (“thoughts”) and taking actions (e.g., calling tools) in a step-by-step or multi-step process.

***

**3. Memory**
- **Memory** enables LangChain applications to “remember” interactions, context, or facts between multiple steps or turns in a conversation. Key types:
    - **BufferMemory:** Stores the full transcript of recent messages.
    - **WindowMemory:** Keeps only the last N messages.
    - **SummarizerMemory:** Periodically summarizes old chat segments.
    - **Custom Memory:** For storing user preferences or facts in specialized formats.
- **Purpose:** LLM calls are stateless by default; memory brings context and continuity.

***

**4. Indexes (Index)**
- **Indexes** connect your LangChain application to external, structured knowledge—PDFs, websites, databases, etc.
- **Types:** Often implemented as vector stores (e.g., FAISS, Pinecone, Chroma) or other retrieval interfaces for efficient searching over large document collections.

***

**5. Chains**
- **Chains** bundle together multiple modular steps (calls to LLMs, retrievers, or other components) into a single, logical pipeline:
    - **Simple Chain:** One input → one output.
    - **Sequential Chain:** Output from one step fed into the next.
    - **Parallel Chain:** Multiple steps run in parallel.
    - **Conditional Chain:** Control flow, like if/else, for branching logic.

***

**6. Prompts**
- **Prompts** are instructions or examples provided to language models to guide their behavior or output.
- **PromptTemplate:** Lets you dynamically insert variables into prompt templates at runtime—making prompts reusable and flexible for many use cases.
- **Types:**
    - **Dynamic/Reusable Prompts**
    - **Role-based Prompts**
    - **Few-shot Prompts**

***

Each of these components can be **combined or customized** for building everything from simple chatbots to advanced retrieval-augmented generation (RAG) systems, knowledge AI assistants, and autonomous AI agents. This modularity is what makes LangChain so powerful for modern GenAI development.

"""
result = chain.invoke({'text': text}) 
print(result)
chain.get_graph().print_ascii()

