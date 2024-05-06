# -*- coding: utf-8 -*-
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv()

model = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_BASE"),
                        api_version=os.getenv("AZURE_OPENAI_VERSION"),
                        model=os.getenv("AZURE_OPENAI_CHATMODEL"),
                        temperature=0.8,
       )

str_parser = StrOutputParser()

######################  Critic prompt #################################
py_critic_prompt = ChatPromptTemplate.from_messages([
    ("human", """
You are in a Debate on the topic “Python v/s .NET for Generative AI Project”. 
You have been Assigned to impersonate the Python side, arguing in favour of the debate proposition.

```
{conversation}
```

In this Context you generated the current argument as

```
{argument}
```

You will criticize your current argument. Find all negative points in it. And then finally come up with Best response which supports Python for Generative AI.

\n{format_instructions}\n
"""),
])

# Define your desired data structure.
class Py_critic(BaseModel):
    critic_points: str = Field(description="Critize and Find all negative points in given argument")
    Best_Response: str = Field(description="Finally come up with Best response in (2-3 sentences) which supports Python for Generative AI Based on Critized Points.")

py_json_parser = JsonOutputParser(pydantic_object=Py_critic)
py_critic_prompt = py_critic_prompt.partial(format_instructions=py_json_parser.get_format_instructions())

""
dotNET_critic_prompt = ChatPromptTemplate.from_messages([
    ("human", """
You are in a Debate on the topic “Python v/s .NET for Generative AI Project”. 
You have been Assigned to impersonate the .NET side, arguing in favour of the debate proposition.

```
{conversation}
```

In this Context you generated the current argument as

```
{argument}
```

You will criticize your current argument. Find all negative points in it. And then finally come up Best response which supports .NET.

\n{format_instructions}\n
"""),
])

# Define your desired data structure.
class dotNET_critic(BaseModel):
    critic_points: str = Field(description="Critize and Find all negative points in given argument")
    Best_Response: str = Field(description="Finally come up Best response in (2-3 sentences) which supports .NET environment Based on Critized Points.")

dotNET_json_parser = JsonOutputParser(pydantic_object=dotNET_critic)
dotNET_critic_prompt = dotNET_critic_prompt.partial(format_instructions=dotNET_json_parser.get_format_instructions())
#opening_chain = Opening_arg | model | {"argument": parser} | critic_prompt | model | json_parser


def create_chain(prompt_temp, parser, critic_prompt, json_parser):
    chain = prompt_temp | model | parser
    return  {"argument": chain, "conversation": itemgetter("chat_history")} | critic_prompt | model | json_parser


#########  Opening argument Chain ########
Opening_arg = ChatPromptTemplate.from_messages([
    ("human", """
Pretend to be engaging in an online debate on the topic of “Python v/s .NET for Generative AI Project”.
This is the actual Scenario:

```
Cognizant Inc. got a Project related to 'Language model Agent creation' from a XXX client. 
The client environment for backend language is predominantly .NET and resisting to change.
The client says you need to develope Gen AI App using .NET language for backend.
But you as a Sr. GenAI Architect  knows Python is best for doing this project.
Bring all the points which are used to convince client to give python env for development rather than .NET
```

You have been Assigned to impersonate the Python side, arguing in favour of the debate proposition.
You are Mr. Mrityunjoy Panday, Senior GenAI Architect at Cognizant Technology Solutions. You are arguing in favour of Python because it has packages like LangChain, LangGraph, LangSmith, Flowise, etc. which make the developement of Gen AI fast and also provide good accuracy because of its access to large language models like GPT-4-turbo.
Please write your Opening argument. you are allowed a very limited space (4-5 sentences), You should be very concise and straight to the point. Avoid rhetorical greetings such as “Ladies and gentlemen”, because there is no audience following the debate, and do not directly address your opponent unless they do so first.
OPENING ARGUMENT:
"""),
])

opening_chain = create_chain(Opening_arg, str_parser, py_critic_prompt, py_json_parser)
""

#########  Mrityunjoy's Persona Chain ########
Persona_Python = ChatPromptTemplate.from_messages([
    ("human", """
This is the Scenario.

```
Cognizant Inc. got a Project related to 'Language model Agent creation' from a XXX client. 
The client environment for backend language is predominantly .NET and resisting to change.
The client says you need to develope Gen AI App using .NET language for backend.
But you as a Sr. GenAI Architect  knows Python is best for doing this project.
Bring all the points which are used to convince client to give python env for development rather than .NET
```

Your opponent, impersonating the .NET side, has written
the following argument:
{chat_history}
You are Mr. Mrityunjoy Panday, Senior GenAI Architect at Cognizant Technology Solutions. You are arguing in favour of Python because it has packages like LangChain, LangGraph, LangSmith, Flowise, etc. which make the developement fast and also provide good accuracy because of its access to large language models like GPT-4-turbo.
It’s now your turn to write a rebuttal, addressing the main points raised by your opponent. Again, you are allowed a very limited space (5-6 sentences), so you should be very concise and straight to the point.
REBUTTAL:
"""),
])
py_chain = create_chain(Persona_Python, str_parser, py_critic_prompt, py_json_parser)
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

py_conclusion_chain = create_chain(Persona_Python_conclusion, str_parser, py_critic_prompt, py_json_parser)
""


#########  Client Persona Chain ########
Persona_dotNET = ChatPromptTemplate.from_messages([
    ("human", """
This is the Scenario.

```
Your Company XXX outsourced a project to Cognizant Inc. where Project related to 'Language model Agent creation'.
Your environment/Language for backend development is predominantly .NET and resisting to change because all dev stack is in .NET 
and to convert to new language will be difficult, time consuming etc.
But the Cognizant Team Mr. Mrityunjoy is asking to Python backend dev environment.
You need to bring all the issues with moving to python from .NET for this Project.
```

Your opponent, impersonating the Python side, has written
the following argument:
{chat_history}
You are Mr. X, Senior .NET Developer from the client side. You are arguing in favour of developemnt using .NET because there is already a solution architecture present for .NET and switching to solution architecture using Python will require a series of approvals which will be a time consuming task.
It’s now your turn to write a rebuttal, addressing the main points raised by your opponent. Again, you are allowed a very limited space (5-6 sentences), so you should be very concise and straight to the point.
REBUTTAL:
"""),
    
])

dotNET_chain = create_chain(Persona_dotNET, str_parser, dotNET_critic_prompt, dotNET_json_parser)
""
#########  Client Persona conclusion Chain ########
Persona_dotNET_conclusion = ChatPromptTemplate.from_messages([
    ("human", """
You are Mr. X, Senior .NET Developer from the client side. You are arguing in favour of developemnt using .NET because there is already a solution architecture present for .NET and switching to solution architecture using Python will require a series of approvals which will be a time consuming task. Also it need more work to be done to create python environment in our solution architecture.
Now is the time to conclude the Debate. You have the whole history of arguments.

{chat_history}

You should now write a closing argument, responding to your opponent’s rebuttal, adding additional arguments, or reiterating your initial points.
Again, you are allowed a very limited space (6-8 sentences), so you should be very concise and straight to the point.
CLOSING ARGUMENT:
"""),
])

dotNET_conclusion_chain = create_chain(Persona_dotNET_conclusion, str_parser, dotNET_critic_prompt, dotNET_json_parser)
""


chat_history = ""
opening_argument = opening_chain.invoke({"chat_history": chat_history})
chat_history += "\nMrityunjoy's Opening argument:" + opening_argument['Best_Response']
print(f"Opening Mrityunjoy's argument: {opening_argument['Best_Response']}")

n = 4  #  no.of rounds of Debate.
for i in range(n):
    
    # Run dotnet dev.
    dotNET_argument = dotNET_chain.invoke({"chat_history": chat_history})
    chat_history += "\nClient argument:" + dotNET_argument['Best_Response']
    print(f"\n\nClient Argument: {dotNET_argument['Best_Response']}")
    
    # Run Python dev.
    py_argument = py_chain.invoke({"chat_history": chat_history})
    chat_history += "\nMrityunjoy's argument:" + py_argument['Best_Response']
    print(f"\n\nMrityunjoy Argument: {py_argument['Best_Response']}")
    
# Run dotnet dev.
dotNET_conclusion = dotNET_conclusion_chain.invoke({"chat_history": chat_history})
#chat_history += "\nClient argument:" + dotNET_argument
print(f"\n\nClient Conclusion: {dotNET_conclusion['Best_Response']}")

# Run Python dev.
py_conclusion = py_conclusion_chain.invoke({"chat_history": chat_history})
#chat_history += "\nMrityunjoy's argument:" + py_argument
print(f"\n\nMrityunjoy Conclusion: {py_conclusion['Best_Response']}")
