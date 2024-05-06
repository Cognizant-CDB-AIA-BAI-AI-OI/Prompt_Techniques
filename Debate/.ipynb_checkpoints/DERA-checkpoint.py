# -*- coding: utf-8 -*-
##  LLM as a Judge included in line 114 decider chat prompt.
##
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
                        max_tokens=4000,
       )

str_parser = StrOutputParser()


def create_chain(prompt_temp, parser):
    return   prompt_temp | model | parser


init_decider_prompt = ChatPromptTemplate.from_messages([
    ("human", """
----
Below is a Debate between a client(arguing for .NET (or) resisting to change to Python env) Vs Mr. Mrityunjoy, Sr. Architect(arguing to create a python env in client Dev env). 
Topic: “Python v/s .NET for Generative AI”.
----
Debate arguments till now
----
{chat_history}
----
Next Best argument
----
Provide the next best argument Supporting "Python".
Do not repeat the already existing points.
The argument must be concise in (2-3 sentences).
----
Best argument
----
"""),
])
initial_chain = create_chain(init_decider_prompt, str_parser)

Researcher_prompt = ChatPromptTemplate.from_messages([
    ("human", """
You (Person B) are a very good argument editer for a Debate between 
Client and Mr. Mrityunjoy on the Topic “Python v/s .NET for Generative AI”.

This is the arguments happened untill now.
-Argument History
{chat_history}
-Argument History

You are discussing the next best argument that another arguer (Person A) 
wrote for this history of arguments.

You will be giving Person A points for correction based on any mistakes
discrepancies you see between the argument history and next best supporting argument.
Person A will add the points of correction that they agree on to a scratchpad to later make edits.

This is Person A's original Version of Argument:
-Person A's Best Supporting Argument-
{argument}
-Person A's Best Supporting Argument-

Here is Person A's current scratchpad of the corrections to make to the summary:
-Corretion Scratchpad-
{scratchpad}
-Corretion Scratchpad-

Go through the argument thoroughly and point out any text that does not have a 
grouding. 

Make sure to make accurate, useful suggestions for corrections.

Person A may not initially agree with you, but if you are confident there is an error do 
your best to convince Person A of the mistake.

Once you have gone through new argument and have confirmed with Person A, and your are satisfied
 with all of the corrections added to the scratchpad and or all of Person A's reasoning to reject 
 additional correction, output the tag "[STOP]".
 
This is the summary discussion with Person A so far:
-Summary Discussion-
{discussion}
-Summary Discussion-

Question: What do you say next? Respond to Person A in the tag [RESPONSE: "<your_response_here>"].
if you are done correctiong and are satisfies, output the "[STOP]" tag.
Answer:
"""),
])
researcher_chain = create_chain(Researcher_prompt, str_parser)

Decider_chat_prompt = ChatPromptTemplate.from_messages([
    ("human", """
You (Person A) are an expert in giving very strong next argument in a debate for given history of arguments.
 
These are the arguments happened until now.
<Argument History>
{chat_history}
</Argument History>
 
You are discussing the argument you wrote for the above history of arguments with another debater(Person B) who verifies your argument for correctness.
 
Person B will suggest some points for correction to the argument. Please act as an impartial judge and evaluate the quality of the argument provided by the researcher(Person B). Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the argument. Begin your evaluation by providing a short explanation. Be as objective as possible. Once the evaluation is done you can add the points of correction to a scratchpad if you agree with them. You also suggest any corrections of your own in case you notice a mistake.
 
This is your original version of Argument:
<Original Argument>
{argument}
</Original Argument>
 
Here is your current scratchpad of the corrections to make to the origina argument:
<Correction Scratchpad>
{scratchpad}
</Correction Scratchpad>
 
This is the summary discussion so far:
<Summary Discussion>
{discussion}
</Summary Discussion>
 
Question: What do you say next? Respond to Person B in the tag [RESPONSE: "<your_response_here>"]
and output any corrections to add to the scratchpad in the tag [SCRATCHPAD: "<things_to_add_to_the_scratchpad_here>"].
Make sure to use the "[]" when outputting tags.
Answer:
"""),
])
Decider_chain = create_chain(Decider_chat_prompt, str_parser)

Decider_final_prompt = ChatPromptTemplate.from_messages([
    ("human", """
You are a very strong Debate next argument creator for given Debate
history of arguments. The argument is between client(arguing for .NET (or) resisting to change to Python env) Vs Mr. Mrityunjoy, Sr. Architect(arguing to create a python env in client Dev env).

This is the Debate arguments till now
-Debate History-
{chat_history}
-Debate History-

This is your original version of the argument:
-Original Argument-
{argument}
-Original Argument-

Here is your current scratchpad of the corrections to make to the origina argument:
-Correction Scratchpad-
{scratchpad}
-Correction Scratchpad-

Make all changes mentioned in the scratchpad to the original argument to output the corrected Best argument.

Output the tag "[STOP]" when finished writing the corrected argument.

The Best argument must be concise in (3-4 sentences).
-Corrected Best Argument-
"""),
])
conclusion_chain = create_chain(Decider_final_prompt, str_parser)



def DERA(chat_history, argument):
    discussion = ""
    scratchpad = ""

    #argument = initial_chain.invoke({"chat_history": chat_history})
    discussion += "Person A:" + argument
    #print(f"Person A: {argument}\n\n-------------------------\n\n")

    n = 4  #  no.of rounds of Debate.
    for i in range(n):

        # Run dotnet dev.
        researcher_suggestion = researcher_chain.invoke({"chat_history": chat_history, 
                                                   "argument": argument, 
                                                   "scratchpad": scratchpad, 
                                                   "discussion": discussion})

        #print(f"\n\nResearcher_responce: {researcher_suggestion}")
        if "[STOP]" not in researcher_suggestion:
            discussion += "\nPerson B:" + researcher_suggestion[12:-2]
            #print(f"\n\nPerson B: {researcher_suggestion[12:-2]}")
        else:
            #print(f"Breaking : {researcher_suggestion}")
            break
        #print("\n\n-------------------------\n\n")

        # Run Python dev.
        decider_response = Decider_chain.invoke({"chat_history": chat_history, 
                                            "argument": argument, 
                                            "scratchpad": scratchpad, 
                                            "discussion": discussion})

        #print(f"\n\ndecider_responce: {decider_response}")
        response_to_personB = decider_response.split("[SCRATCHPAD:")[0][12:]

        discussion += "\nPerson A:" + response_to_personB
        #print(f"\n\nPerson A: {response_to_personB}")
        scratchpad += "\n" + decider_response.split('[SCRATCHPAD:')[1][:-2]
        #print("\n\nCurrent Scratchpad: ", scratchpad)
        #print("\n\n-------------------------\n\n")

        #break
    # Run dotnet dev.
    conclusion = conclusion_chain.invoke({"chat_history": chat_history, 
                                          "argument": argument, 
                                          "scratchpad": scratchpad})
    #print(f"\n\nConclusion: {conclusion}")
    
    return conclusion.replace("[STOP]", "").strip()





"""
You (Person A) are a very strong Debate next argument creator for given Debate
history of arguments.

This is the arguments happened untill now.
-Argument History-
{chat_history}
-Argument History-

You are discussing the Argument you wrote for this History with another Argument creator
(Person B) whose job is to verify your argument for correctness.

Person B will give you points for correction and it will be your job to add the points of correction to a 
Scratchpad if you agree with them.

This is your original version of Argument:
-Original Argument-
{argument}
-Original Argument-

Here is your current scratchpad of the corrections to make to the origina argument:
-Correction Scratchpad-
{scratchpad}
-Correction Scratchpad-

You are generally very confident about the argument you wrote,
however, when presented with compelling arguments by the verifying arugment editor, you add to the correction scratchpad.
You also suggest any edits of your own in case you notice a mistake.

This is the summary discussion so far:
-Summary Discussion-
{discussion}
-Summary Discussion-

Question: What do you say next? Respond to Person B in the tag [RESPONSE: "<your_response_here>"]
and output any corrections to add to the scratchpad in the tag [SCRATCHPAD: "<things_to_add_to_the_scratchpad_here>"].
Make sure to use the "[]" when outputting tags.
Answer:
"""
