import os
from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import chainlit as cl
from typing import cast
from dotenv import load_dotenv
from agents.run import RunConfig
from career_agents import career_agent

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("GEMINI_BASE_URL")
MODEL = "gemini-2.0-flash"

@cl.on_chat_start
async def chat_start():
    client = AsyncOpenAI(api_key= gemini_api_key, base_url= BASE_URL)

    config = RunConfig(model=OpenAIChatCompletionsModel(model= MODEL, openai_client= client),
                       model_provider= client,
                       tracing_disabled= True)
    
    cl.user_session.set("config", config)
    cl.user_session.set("agent", career_agent)
    cl.user_session.set("chathistory",[])

    await cl.Message(f"""Welcome to the Career Explorer👋 I'm here to help you find your path.
                     What courses have you taken? Tell me about your skills.""").send()
    
@cl.on_message
async def main(message:cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    agent:Agent = cast(Agent,cl.user_session.get("agent"))
    config = cast(RunConfig,cl.user_session.get("config"))
    history = cl.user_session.get("chathistory") or []
    history.append({"role":"user","content":message.content})
 
    try:
        print("\nCALLING AGENT WITH CONTEXT\n",history,"\n")
        result = await Runner.run(starting_agent= agent, input= history, run_config= config)
        msg.content = result.final_output
        await msg.update()
        cl.user_session.set("chathistory",result.to_input_list())
        print(f"User: {message.content}")
        print(f"AI Assistant: {msg.content}")
        
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")

    
    
