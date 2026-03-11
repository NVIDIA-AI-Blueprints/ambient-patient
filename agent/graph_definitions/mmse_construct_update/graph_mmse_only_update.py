# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dotenv import load_dotenv
import os
import logging

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_neo4j import Neo4jSaver
from neo4j import GraphDatabase
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langchain.messages import AnyMessage

from langfuse import get_client
from langfuse.langchain import CallbackHandler

import sys 
from typing import Annotated
from typing_extensions import TypedDict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.logging import get_logging
from graph_definitions.utils_graph import add_messages_with_reset

#################
### variables ###
#################

env_var_file = "/env_vars/vars.env"

def load_env_variables():
    """Load environment variables from vars.env file and return configuration"""
    #########################
    ### get env variables ###
    #########################
    load_dotenv(env_var_file, override=True)  # This line brings all environment variables from vars.env into os.environ

    assert os.environ['NVIDIA_API_KEY'] is not None, "Make sure you have your NVIDIA_API_KEY exported as a environment variable!"
    nvidia_api_key = os.getenv("NVIDIA_API_KEY", None)

    assert os.environ['AGENT_LLM_MODEL'] is not None, "Make sure you have your AGENT_LLM_MODEL exported as a environment variable!"
    llm_model = os.getenv("AGENT_LLM_MODEL", None)

    assert os.environ['AGENT_LLM_BASE_URL'] is not None, "Make sure you have your AGENT_LLM_BASE_URL exported as a environment variable!"
    base_url = os.getenv("AGENT_LLM_BASE_URL", None)

    nemo_guardrails_config_path = os.getenv("NEMO_GUARDRAILS_CONFIG_PATH", None)

    # Langfuse credentials
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", None)
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", None)
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL", None)
    langfuse_prompt_name = os.getenv("LANGFUSE_PROMPT_NAME", "patient-intake-initial")
    langfuse_prompt_label = os.getenv("LANGFUSE_PROMPT_LABEL", "production")

    # Neo4j credentials
    neo4j_uri = os.getenv("NEO4J_URI", "").strip()
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    neo4j_password = os.getenv("NEO4J_PASSWORD", "").strip()

    return {
        'nvidia_api_key': nvidia_api_key,
        'llm_model': llm_model,
        'base_url': base_url,
        'nemo_guardrails_config_path': nemo_guardrails_config_path,
        'langfuse_public_key': langfuse_public_key,
        'langfuse_secret_key': langfuse_secret_key,
        'langfuse_base_url': langfuse_base_url,
        'langfuse_prompt_name': langfuse_prompt_name,
        'langfuse_prompt_label': langfuse_prompt_label,
        'neo4j_uri': neo4j_uri,
        'neo4j_username': neo4j_username,
        'neo4j_password': neo4j_password,
    }
# enable logging
logger = get_logging(__name__)

########################
### Define the tools ###
########################

#@tool
#def print_gathered_patient_info():
#    pass


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages_with_reset]

class MmseAgent:
    def __init__(self, model, prompts):
        self.agent = create_agent(
            model=model, 
            tools=[], 
            system_prompt=prompts,
            middleware=[ToolRetryMiddleware()]) # AgentMiddleware() to add later
        
    async def ainvoke(self, state: State, config: RunnableConfig):
        return self.agent.ainvoke(state, config)
    
    def invoke(self, state: State, config: RunnableConfig):
        return self.agent.invoke(state, config)

def create_mmse_graph():
    """Create and return a fresh instance of the intake graph"""
    # Reload environment variables from vars.env
    env_config = load_env_variables()
    
    # Create fresh LLM instance with reloaded config
    assistant_llm = ChatNVIDIA(
        model=env_config['llm_model'], 
        base_url=env_config['base_url']
    )
    # Open a langfuse connection
    langfuse = get_client()
    langfuse_handler = CallbackHandler()
    
    # Get langfuse prompts
    agent_prompts = langfuse.get_prompt(env_config['langfuse_prompt_name']).prompt
    logging.info(f"{agent_prompts[0]['content']}")

    # Create a fresh StateGraph builder
    builder = StateGraph(State)
    
    # Define nodes: these do the work
    builder.add_node("patient_intake_assistant", MmseAgent(assistant_llm, agent_prompts[0]['content']).invoke)
    
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "patient_intake_assistant")
    builder.add_edge("patient_intake_assistant", END)
    
    # Create a Neo4j checkpointer from an explicit driver so the connection
    # stays open after this function returns (the `with` context manager would
    # close it on exit, making the returned graph unusable).
    driver = GraphDatabase.driver(
        env_config['neo4j_uri'],
        auth=(env_config['neo4j_username'], env_config['neo4j_password']),
    )
    checkpointer = Neo4jSaver(driver)
    checkpointer.setup()  # Create indexes (run once)
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Graph built successfully")
    return graph
