# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Tuple
import uuid
import sys 
import os
import logging
import gradio as gr
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from ui_assets.css.css import css, theme
from utils.stream import print_event_async_stream

log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)

logger = logging.getLogger(__name__)

def get_config_with_new_thread_id():
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
            "recursion_limit": 10,
        }
    }
    logger.info("New config created with thread id: " + thread_id)
    return config



def launch_demo_ui(assistant_graph):
    # Establish a connection to the Riva server
   
    global config 
    config = get_config_with_new_thread_id()
    async def interact(query: str, chat_history: List[Tuple[str, str]], full_response: str):
        last_answer, added_log = await print_event_async_stream(assistant_graph, query, config, max_length=1500)
        updated_full_response = full_response + added_log if full_response else added_log
        yield "", chat_history + [[query, last_answer]], updated_full_response, last_answer

    def new_thread():
        global config
        config = get_config_with_new_thread_id()
        logger.info("Updated new thread id in config: " + config["configurable"]["thread_id"])


    with gr.Blocks(title = "Welcome to the Healthcare Contact Center", theme=theme, css=css) as demo:
        gr.Markdown("# Welcome to the Healthcare Contact Center")

        # session specific state across runs
        state = gr.State()
        

        with gr.Row(equal_height=True):
            chatbot = gr.Chatbot(label="Healthcare Contact Center Agent", elem_id="chatbot")
            latest_response = gr.Textbox(label = "Latest Response", visible=False)
            full_response = gr.Textbox(label = "Full Response Log", visible=False, elem_id="fullresponsebox", lines=25)
    

        
        # input
        with gr.Row():
            with gr.Column(scale = 10):
                msg = gr.Textbox(label = "Input Query", show_label=True, placeholder="Enter text and press ENTER", container=False,)
        
        # buttons
        with gr.Row():
            submit_btn = gr.Button(value="Submit")
            _ = gr.ClearButton([msg, chatbot], value="Clear UI")
            full_resp_show = gr.Button(value="Show Full Response")
            full_resp_hide = gr.Button(value="Hide Full Response", visible=False)
        with gr.Row():
            new_thread_btn = gr.Button(value="Clear Chat Memory")
        
        # hide/show context
        def _toggle_full_response(btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Show Full Response":
                out = [True, False, True]
            if btn == "Hide Full Response":
                out = [False, True, False]
            return {
                full_response: gr.update(visible=out[0]),
                full_resp_show: gr.update(visible=out[1]),
                full_resp_hide: gr.update(visible=out[2]),
            }

        full_resp_show.click(_toggle_full_response, [full_resp_show], [full_response, full_resp_show, full_resp_hide])
        full_resp_hide.click(_toggle_full_response, [full_resp_hide], [full_response, full_resp_show, full_resp_hide])

        msg.submit(interact, [msg, chatbot, full_response], [msg, chatbot, full_response, latest_response])
        submit_btn.click(interact, [msg, chatbot, full_response], [msg, chatbot, full_response, latest_response])

        new_thread_btn.click(new_thread)

    return demo
    