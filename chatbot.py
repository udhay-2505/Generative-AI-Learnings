from langchain_groq import ChatGroq
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

model= ChatGroq(api_key= os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile" )

chat_history=[
    SystemMessage(content='you are a helpful AI assistant')
]
while True:
    user_input= input("You:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result= model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

#print(chat_history)
         