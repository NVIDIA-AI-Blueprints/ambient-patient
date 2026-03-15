# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice Agent WebRTC Pipeline.

This module implements a voice agent pipeline using WebRTC for real-time
speech-to-speech communication with dynamic prompt support.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
import httpx
import uvicorn
import yaml
from config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import InputAudioRawFrame, LLMMessagesFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import (
    IceServer,
    SmallWebRTCConnection,
)
from websocket_transcript_output import WebsocketTranscriptOutput

from nvidia_pipecat.processors.audio_util import AudioRecorder
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.nvidia_rag import NvidiaRAGService
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService

load_dotenv(override=True)

config_path = os.getenv("CONFIG_PATH")
if not config_path:
    raise ValueError("CONFIG_PATH environment variable is not set")
try:
    raw_yaml = os.path.expandvars(Path(config_path).read_text())
    unresolved = re.findall(r'\$\{[^}]+\}', raw_yaml)
    if unresolved:
        raise ValueError(
            f"Unresolved environment variables in config file {config_path}: {unresolved}"
        )
    config = Config(**yaml.safe_load(raw_yaml))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Config file not found at: {config_path}") from e
except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML in config file: {e}") from e


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connections by pc_id
pcs_map: dict[str, SmallWebRTCConnection] = {}
contexts_map: dict[str, OpenAILLMContext] = {}


ice_servers = (
    [
        IceServer(
            urls=os.getenv("TURN_SERVER_URL", ""),
            username=os.getenv("TURN_USERNAME", ""),
            credential=os.getenv("TURN_PASSWORD", ""),
        )
    ]
    if os.getenv("TURN_SERVER_URL")
    else []
)


async def run_bot(webrtc_connection, ws: WebSocket):
    """Run the voice agent bot with WebRTC connection and WebSocket.

    Args:
        webrtc_connection: The WebRTC connection for audio streaming
        ws: WebSocket connection for communication
    """
    stream_id = uuid.uuid4()
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=16000,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        audio_out_10ms_chunks=5,
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=transport_params,
    )

    agent = NvidiaRAGService(
        collection_name=config.NvidiaRAGService.collection_name,
        enable_citations=config.NvidiaRAGService.enable_citations,
        rag_server_url=config.NvidiaRAGService.rag_server_url,
        use_knowledge_base=config.NvidiaRAGService.use_knowledge_base,
        max_tokens=config.NvidiaRAGService.max_tokens,
        filler=config.Pipeline.filler,
        session = httpx.AsyncClient(timeout=float(os.getenv("REQUEST_TIMEOUT", 15.0)))
    )

    stt = RivaASRService(
        server=config.RivaASRService.server,
        api_key=os.getenv("NVIDIA_API_KEY"),
        language=config.RivaASRService.language,
        sample_rate=config.RivaASRService.sample_rate,
        automatic_punctuation=True,
        model=config.RivaASRService.model,
        function_id=config.RivaASRService.function_id,
    )

    # Load IPA dictionary with error handling
    ipa_file = Path(__file__).parent / "ipa.json"
    try:
        with open(ipa_file, encoding="utf-8") as f:
            ipa_dict = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"IPA dictionary file not found at {ipa_file}")
        raise FileNotFoundError(f"IPA dictionary file not found at {ipa_file}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in IPA dictionary file: {e}")
        raise ValueError(f"Invalid JSON in IPA dictionary file: {e}") from e
    except Exception as e:
        logger.error(f"Error loading IPA dictionary: {e}")
        raise

    tts = RivaTTSService(
        server=config.RivaTTSService.server,
        api_key=os.getenv("NVIDIA_API_KEY"),
        voice_id=config.RivaTTSService.voice_id,
        model=config.RivaTTSService.model,
        function_id=config.RivaTTSService.function_id,
        language=config.RivaTTSService.language,
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT")) if os.getenv("ZERO_SHOT_AUDIO_PROMPT") else None
        ),
        ipa_dict=ipa_dict,
    )
    # by default, audio recording is disabled, 
    # if you want to record audio, set the environment variable DUMP_AUDIO_FILES to true in ace_controller.env file
    enable_audio_recording = os.getenv("DUMP_AUDIO_FILES", "false").lower() == "true"
    if enable_audio_recording:
        # Create audio_dumps directory if it doesn't exist
        audio_dumps_dir = Path(__file__).parent / "audio_dumps"
        audio_dumps_dir.mkdir(exist_ok=True)

        asr_recorder = AudioRecorder(
            output_file=str(audio_dumps_dir / f"asr_recording_{stream_id}.wav"),
            params=transport_params,
            frame_type=InputAudioRawFrame,
        )

        tts_recorder = AudioRecorder(
            output_file=str(audio_dumps_dir / f"tts_recording_{stream_id}.wav"),
            params=transport_params,
            frame_type=TTSAudioRawFrame,
        )
    else:
        asr_recorder = None
        tts_recorder = None

    # Used to synchronize the user and bot transcripts in the UI
    stt_transcript_synchronization = UserTranscriptSynchronization()
    tts_transcript_synchronization = BotTranscriptSynchronization()

    messages = [
        
    ]

    context = OpenAILLMContext(messages)

    # Store context globally so WebSocket can access it
    pc_id = webrtc_connection.pc_id
    contexts_map[pc_id] = context

    # Configure speculative speech processing based on environment variable
    # set this variable to true only if your agent backend does not retain every incoming request and the agent response in memory
    # we will keep this set to false since the healthcare agent retains memory in langgraph
    enable_speculative_speech = os.getenv("ENABLE_SPECULATIVE_SPEECH", "false").lower() == "true"

    if enable_speculative_speech:
        context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)
        tts_response_cacher = NvidiaTTSResponseCacher()
    else:
        context_aggregator = agent.create_context_aggregator(context)
        tts_response_cacher = None

    transcript_processor_output = WebsocketTranscriptOutput(ws)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            *([asr_recorder] if asr_recorder else []),  # Include asr_recorder only if enabled
            stt,  # Speech-To-Text
            stt_transcript_synchronization,
            context_aggregator.user(),
            agent,  # Agent Backend
            tts,  # Text-To-Speech
            *([tts_recorder] if tts_recorder else []),  # Include tts_recorder only if enabled
            *([tts_response_cacher] if tts_response_cacher else []),  # Include cacher only if enabled
            tts_transcript_synchronization,
            transcript_processor_output,
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=True,
            start_metadata={"stream_id": stream_id},
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Wait 50ms for custom prompt from UI before starting conversation
        await asyncio.sleep(0.05)
        # Kick off the conversation.
        # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for handling voice agent connections.

    Args:
        websocket: The WebSocket connection to handle
    """
    await websocket.accept()
    try:
        request = await websocket.receive_json()
        pc_id = request.get("pc_id")

        if pc_id and pc_id in pcs_map:
            pipecat_connection = pcs_map[pc_id]
            logger.info(f"Reusing existing connection for pc_id: {pc_id}")
            await pipecat_connection.renegotiate(sdp=request["sdp"], type=request["type"])
        else:
            pipecat_connection = SmallWebRTCConnection(ice_servers)
            await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

            @pipecat_connection.event_handler("closed")
            async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                pcs_map.pop(webrtc_connection.pc_id, None)  # Remove connection reference
                contexts_map.pop(webrtc_connection.pc_id, None)  # Remove context reference

            asyncio.create_task(run_bot(pipecat_connection, websocket))

        answer = pipecat_connection.get_answer()
        pcs_map[answer["pc_id"]] = pipecat_connection

        await websocket.send_json(answer)

        # Keep the connection open and print text messages
        while True:
            try:
                message = await websocket.receive_text()
                # Parse JSON message from UI
                try:
                    data = json.loads(message)
                    message = data.get("message", "").strip()
                    if data.get("type") == "context_reset" and message:
                        print(f"Received context reset from UI: {message}")
                        logger.info(f"Context reset from UI: {message}")

                        # Replace entire conversation context with new system prompt
                        pc_id = pipecat_connection.pc_id
                        if pc_id in contexts_map:
                            context = contexts_map[pc_id]
                            context.set_messages([{"role": "system", "content": message}])
                        else:
                            print(f"No context found for pc_id: {pc_id}")

                except json.JSONDecodeError:
                    print(f"Non-JSON message: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from websocket")


@app.get("/get_prompt")
async def get_prompt():
    """Get the default system prompt."""
    return {
        "prompt": "Not set in ace-controller, set in agent backend",
        "name": "System Prompt",
        "description": "Default system prompt for the System as set at the backend",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: localhost)")
    parser.add_argument("--port", type=int, default=7860, help="Port for HTTP server (default: 7860)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)
