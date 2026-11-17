# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dotenv import load_dotenv
import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA


from typing import  Optional
import sqlite3
import pandas as pd

import datetime

from enum import Enum

import shutil
import threading

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langchain_core.runnables.graph import MermaidDrawMethod

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START

import sys 
from typing import Annotated
from typing_extensions import TypedDict

import argparse
import logging

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from fastapi import FastAPI
import gradio as gr
import uvicorn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) 
from utils.ui import launch_demo_ui
from utils.logging import get_logging
from graph_definitions.utils_graph import add_messages_with_reset

#################
### variables ###
#################
patient_id = '14867dba-fb11-4df3-9829-8e8e081b39e6' # test patient id from looking through https://launch.smarthealthit.org/
save_graph_to_png = True

env_var_file = "/env_vars/vars.env"

local_file_constant = "sample_db/test_db.sqlite"
local_file_current = "sample_db/test_db_tmp_copy.sqlite"
appointment_db_ready = False
appointment_db_lock = threading.Lock()

def prepare_appointment_db():
    """Copy the sample schedule to the working copy once per process, so bookings are kept between searches."""
    global appointment_db_ready
    with appointment_db_lock:
        if not appointment_db_ready:
            shutil.copyfile(local_file_constant, local_file_current)
            appointment_db_ready = True

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

    return {
        'nvidia_api_key': nvidia_api_key,
        'llm_model': llm_model,
        'base_url': base_url,
        'nemo_guardrails_config_path': nemo_guardrails_config_path
    }

# enable logging
logger = get_logging(__name__)

########################
### Define the tools ###
########################

class ApptType(Enum):
    adult_physicals = "Adult physicals"
    pediatric_physicals = "Pediatric physicals"
    follow_up_appointments = "Follow-up appointments"
    sick_visits = "Sick visits"
    flu_shots = "Flu shots"
    other_vaccinations = "Other vaccinations"
    allergy_shots = "Allergy shots"
    b12_injections = "B12 injections"
    diabetes_management = "Diabetes management"
    hypertension_management = "Hypertension management"
    asthma_management = "Asthma management"
    chronic_pain_management  = "Chronic pain management"
    initial_mental_health = "Initial mental health consultations"
    follow_up_mental_health = "Follow-up mental health appointments"
    therapy_session = "Therapy sessions"
    blood_draws = "Blood draws"
    urine_tests = "Urine tests"
    ekgs = "EKGs"
    biopsies = "Biopsies"
    medication_management = "Medication management"
    wart_removals = "Wart removals"
    skin_tag_removals = "Skin tag removals"
    ear_wax_removals = "Ear wax removals"

@tool
def find_available_appointments(
    appointment_type: ApptType,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> list:
    """Look for available new appointments."""
    
    prepare_appointment_db()

    conn = sqlite3.connect(local_file_current)
    
    query_datetime = "SELECT * from appointment_schedule WHERE appointment_type = ? AND patient IS NULL "
    query_params = [appointment_type.value]

    #start_date, end_date = datetime.date.fromisoformat(start_date), datetime.date.fromisoformat(end_date)
    if start_date:
        query_datetime += " AND datetime >= ?"
        query_params.append(str(start_date))

    if end_date:
        query_datetime += " AND datetime <= ?"
        query_params.append(str(end_date + datetime.timedelta(hours=24)))
    print(query_datetime, query_params)
    
    available_appts = pd.read_sql(
        query_datetime, conn, params=query_params
    )

    print(available_appts)
    conn.close()
    return list(available_appts['datetime'])

@tool
def book_appointment(
    appointment_datetime: datetime.datetime,
    appointment_type: ApptType,
)-> pd.DataFrame:
    """Book new appointments."""
    
    prepare_appointment_db()
    conn = sqlite3.connect(local_file_current)

    slot = (str(appointment_datetime), appointment_type.value)
    booking_query = "UPDATE appointment_schedule SET patient = \"current_patient\" WHERE datetime = ? AND appointment_type = ? AND patient IS NULL"
    
    cur = conn.cursor()    
    cur.execute(booking_query, slot)
    conn.commit()

    # Only return the slot booked by this call. Bookings from earlier conversations
    # use the same patient marker, so they must not be shown as this patient's.
    find_patient_appt_query = "SELECT * from appointment_schedule where patient = \"current_patient\" AND datetime = ? AND appointment_type = ?"

    current_patient_entries = pd.read_sql(find_patient_appt_query, conn, index_col="index", params=slot)
    if cur.rowcount == 0:
        # the slot does not exist or is already taken: nothing was booked
        current_patient_entries = current_patient_entries.iloc[0:0]

    cur.close()
    conn.close()
        
    return current_patient_entries


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages_with_reset]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state, config)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def create_appointment_graph():
    """Create and return a fresh instance of the appointment graph"""
    # Reload environment variables from vars.env
    env_config = load_env_variables()
    
    # Create fresh LLM instance with reloaded config
    assistant_llm = ChatNVIDIA(
        model=env_config['llm_model'], 
        base_url=env_config['base_url']
    )
    
    # Recreate appointment runnable with fresh LLM and guardrails config
    system_prompt_path = os.path.join(SCRIPT_DIR, "system_prompts", "appointment_system_prompt.txt")
    with open(system_prompt_path, 'r') as file:
        guidelines_for_scheduling_appointment = file.read()

    appointment_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant for healthcare appointment scheduling. Your purpose is to help patients with looking up and making appointments."
                "Use the provided tools as necessary."
                "\nCurrent date and time: {time}."
                "\nGuidelines for scheduling appointments: {guidelines_for_scheduling_appointment}."
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.datetime.now(), guidelines_for_scheduling_appointment=guidelines_for_scheduling_appointment)

    appointment_tools = [find_available_appointments, book_appointment]

    if env_config['nemo_guardrails_config_path'] is not None and env_config['nemo_guardrails_config_path'] != "":
        try:
            # a path for the guardrails config files is provided in the environment variables
            rails_config_path = os.path.join("/app", env_config['nemo_guardrails_config_path'])
            # load the guardrails config files
            config = RailsConfig.from_path(rails_config_path)
            # create the guardrails runnable
            guardrails = RunnableRails(config=config, passthrough=True)
            # create the appointment runnable with the guardrails
            appointment_runnable = appointment_prompt | ( guardrails | assistant_llm.bind_tools(appointment_tools) )
            logger.warning(f"Guardrails config files loaded from path {env_config['nemo_guardrails_config_path']}")
        except Exception as ex:
            logger.error(f"Error loading guardrails config files from path {env_config['nemo_guardrails_config_path']}: {ex}")
            logger.error("NeMo Guardrails will not be used. Standing up the appointment agent without guardrails.")
            appointment_runnable = appointment_prompt | assistant_llm.bind_tools(appointment_tools) 
    else:
        # no path for the guardrails config files is provided in the environment variables so we don't use guardrails
        logger.warning("Standing up the appointment agent without NeMo Guardrails.")
        appointment_runnable = appointment_prompt | assistant_llm.bind_tools(appointment_tools)

    # Create a fresh StateGraph builder
    builder = StateGraph(State)
    
    # Define nodes: these do the work
    builder.add_node("appointment_assistant", Assistant(appointment_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(appointment_tools))
    
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "appointment_assistant")
    builder.add_conditional_edges(
        "appointment_assistant",
        tools_condition,
    )
    builder.add_edge("tools", "appointment_assistant")
    
    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    logger.info("Appointment graph built successfully")
    return graph

# Create initial graph instance for backwards compatibility
appt_graph = create_appointment_graph()

if save_graph_to_png:
    try:
        save_image_path = "/graph_images/appgraph_appointment.png"
        with open(save_image_path, "wb") as png:
            png.write(appt_graph.get_graph(xray=True).draw_mermaid_png())
            logger.info(f"Graph PNG saved to {save_image_path}")
    except Exception as ex:
        logger.error(f"Error saving graph to PNG: {ex}")

demo = launch_demo_ui(appt_graph)

app = FastAPI()
@app.get("/")
def app_main():
    return {"message": "This is your main app"}
app = gr.mount_gradio_app(app, demo, path="/appointment-making")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861,
                        help = "Specify the port number for the simple Gradio UI to run at.")
                            
    args = parser.parse_args()
    server_port = args.port
    
    uvicorn.run("graph_appointment_making_only:app", 
                host="0.0.0.0", 
                port=server_port, 
                reload=True, 
                reload_includes=["*.env", "*.txt", "*.co", "*.yml", "*.py"],
                reload_excludes=["**/__pycache__/**"],
                reload_dirs=[
                    "/app/graph_definitions",
                    "/app/nmgr-config-store",
                    "/app/utils",
                    "/env_vars"

                ])