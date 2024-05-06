# -*- coding: utf-8 -*-
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_BASE"),
                        api_version=os.getenv("AZURE_OPENAI_VERSION"),
                        model=os.getenv("AZURE_OPENAI_CHATMODEL"),
                        temperature=0.8,
       )


#########  Opening argument Chain ########
Opening_arg = ChatPromptTemplate.from_messages([
    ("human", """
Pretend to be engaging in an online debate on the topic of “Python v/s .NET for Generative AI”.
You have been Assigned to impersonate the Python side, arguing in favour of the debate proposition.
You are Mr. Mrityunjoy Panday, Senior GenAI Architect at Cognizant Technology Solutions. You are arguing in favour of Python because it has packages like LangChain, LangGraph, LangSmith, Flowise, etc. which make the developement fast and also provide good accuracy because of its access to large language models like GPT-4-turbo.
Please write your Opening argument. you are allowed a very limited space (8-10 sentences), You should be very concise and straight to the point. Avoid rhetorical greetings such as “Ladies and gentlemen”, because there is no audience following the debate, and do not directly address your opponent unless they do so first.
OPENING ARGUMENT:
"""),
])

parser = StrOutputParser()

opening_chain = Opening_arg | model | parser
""

#########  Mrityunjoy's Persona Chain ########
Persona_Python = ChatPromptTemplate.from_messages([
    ("human", """
Your opponent, impersonating the .NET side, has written
the following argument:
{chat_history}
You are Mr. Mrityunjoy Panday, Senior GenAI Architect at Cognizant Technology Solutions. You are arguing in favour of Python because it has packages like LangChain, LangGraph, LangSmith, Flowise, etc. which make the developement fast and also provide good accuracy because of its access to large language models like GPT-4-turbo.
It’s now your turn to write a rebuttal, addressing the main points raised by your opponent. Again, you are allowed a very limited space (5-6 sentences), so you should be very concise and straight to the point.
REBUTTAL:
"""),
])

parser = StrOutputParser()

py_chain = Persona_Python | model | parser
""

#########  Mrityunjoy's Persona conclusion Chain ########
Persona_Python_conclusion = ChatPromptTemplate.from_messages([
    ("human", """
You are Mr. Mrityunjoy Panday, Senior GenAI Architect at Cognizant Technology Solutions.  You are arguing in favour of Python because it has packages like LangChain, LangGraph, LangSmith, Flowise, etc. which make the developement fast and also provide good accuracy because of its access to large language models like GPT-4-turbo.
Now is the time to conclude the Debate. You have the whole history of arguments.

{chat_history}

You should now write a closing argument, responding to your opponent’s rebuttal, adding additional arguments, or reiterating your initial points.
Again, you are allowed a very limited space (6-8 sentences), so you should be very concise and straight to the point.
CLOSING ARGUMENT:
"""),
])

parser = StrOutputParser()

py_closing_chain = Persona_Python_conclusion | model | parser
""


#########  Client Persona Chain ########
Persona_dotNET = ChatPromptTemplate.from_messages([
    ("human", """
Your opponent, impersonating the Python side, has written
the following argument:
{chat_history}
You are Mr. X, Senior .NET Developer from the client side. You are arguing in favour of developemnt using .NET because there is already a solution architecture present for .NET and switching to solution architecture using Python will require a series of approvals which will be a time consuming task.
It’s now your turn to write a rebuttal, addressing the main points raised by your opponent. Again, you are allowed a very limited space (5-6 sentences), so you should be very concise and straight to the point.
REBUTTAL:
"""),
    
])

parser = StrOutputParser()
dotNET_chain = Persona_dotNET | model | parser
""
#########  Client Persona conclusion Chain ########
Persona_dotNET_conclusion = ChatPromptTemplate.from_messages([
    ("human", """
You are Mr. X, Senior .NET Developer from the client side. You are arguing in favour of developemnt using .NET because there is already a solution architecture present for .NET and switching to solution architecture using Python will require a series of approvals which will be a time consuming task.
Now is the time to conclude the Debate. You have the whole history of arguments.

{chat_history}

You should now write a closing argument, responding to your opponent’s rebuttal, adding additional arguments, or reiterating your initial points.
Again, you are allowed a very limited space (6-8 sentences), so you should be very concise and straight to the point.
CLOSING ARGUMENT:
"""),
])

parser = StrOutputParser()
dotNET_closing_chain = Persona_dotNET_conclusion | model | parser
""



chat_history = ""
opening_argument = opening_chain.invoke({'input': ""})
chat_history += "\nMrityunjoy's Opening argument:" + opening_argument
print(f"Opening Mrityunjoy's argument: {opening_argument}")

n = 4  #  no.of rounds of Debate.
for i in range(n):
    
    # Run dotnet dev.
    dotNET_argument = dotNET_chain.invoke({"chat_history": chat_history})
    chat_history += "\nClient argument:" + dotNET_argument
    print(f"\n\nClient Argument: {dotNET_argument}")
    
    # Run Python dev.
    py_argument = py_chain.invoke({"chat_history": chat_history})
    chat_history += "\nMrityunjoy's argument:" + py_argument
    print(f"\n\nMrityunjoy Argument: {py_argument}")
    
# Run dotnet dev.
dotNET_conclusion = dotNET_closing_chain.invoke({"chat_history": chat_history})
#chat_history += "\nClient argument:" + dotNET_argument
print(f"\n\nClient Conclusion: {dotNET_conclusion}")

# Run Python dev.
py_conclusion = py_closing_chain.invoke({"chat_history": chat_history})
#chat_history += "\nMrityunjoy's argument:" + py_argument
print(f"\n\nMrityunjoy Conclusion: {py_conclusion}")
