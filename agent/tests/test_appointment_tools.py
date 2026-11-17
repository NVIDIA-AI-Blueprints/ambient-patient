# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the appointment tools (find_available_appointments, book_appointment).

The graph modules build their LLM graph and UI at import time, so the model, FHIR,
Tavily, guardrails and web server dependencies are replaced with stubs here. Only
langchain-core, langgraph, pandas and pytest are needed:

    pip install langchain==0.3.27 langgraph==0.6.6 pandas==2.2.2 pytest
    cd agent && python -m pytest tests
"""

import datetime
import importlib
import logging
import os
import sqlite3
import sys
import types

import pytest

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAPH_MODULES = [
    "graph_definitions.graph",
    "graph_definitions.graph_appointment_making_only",
]

SLOTS = [
    (0, "2025-10-07 09:00:00", "Doctor A", "Flu shots"),
    (1, "2025-10-07 16:00:00", "Doctor B", "Flu shots"),
    (2, "2025-10-08 10:00:00", "Doctor A", "Flu shots"),
    (3, "2025-10-08 11:00:00", "Doctor C", "Adult physicals"),
]


def _stub_modules():
    """Stand-ins for the dependencies that need API keys, network access or a UI."""
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool

    stubs = {}
    for name in [
        "dotenv",
        "langchain_nvidia_ai_endpoints",
        "fhirclient",
        "fhirclient.client",
        "fhirclient.models",
        "fhirclient.models.patient",
        "fhirclient.models.medication",
        "fhirclient.models.medicationrequest",
        "langchain_community",
        "langchain_community.tools",
        "langchain_community.tools.tavily_search",
        "nemoguardrails",
        "nemoguardrails.integrations",
        "nemoguardrails.integrations.langchain",
        "nemoguardrails.integrations.langchain.runnable_rails",
        "gradio",
        "uvicorn",
        "fastapi",
        "utils.ui",
    ]:
        stubs[name] = types.ModuleType(name)

    class ChatNVIDIA:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            return RunnableLambda(lambda x: x)

    @tool
    def search_stub(query: str) -> str:
        """Stand-in for TavilySearchResults."""
        return ""

    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda f: f

    stubs["dotenv"].load_dotenv = lambda *args, **kwargs: None
    stubs["langchain_nvidia_ai_endpoints"].ChatNVIDIA = ChatNVIDIA
    stubs["fhirclient.client"].FHIRClient = lambda settings: types.SimpleNamespace(
        server=None
    )
    stubs["fhirclient"].client = stubs["fhirclient.client"]
    stubs["fhirclient.models.patient"].Patient = object
    stubs["fhirclient.models.medication"].Medication = object
    stubs["fhirclient.models.medicationrequest"].MedicationRequest = object
    stubs["langchain_community.tools.tavily_search"].TavilySearchResults = (
        lambda **kwargs: search_stub
    )
    stubs["nemoguardrails"].RailsConfig = object
    stubs["nemoguardrails.integrations.langchain.runnable_rails"].RunnableRails = object
    stubs["fastapi"].FastAPI = FastAPI
    stubs["gradio"].mount_gradio_app = lambda *args, **kwargs: None
    stubs["utils.ui"].launch_demo_ui = lambda graph: None
    return stubs


@pytest.fixture(scope="module", params=GRAPH_MODULES)
def graph_module(request):
    with pytest.MonkeyPatch.context() as mp:
        for name, module in _stub_modules().items():
            mp.setitem(sys.modules, name, module)
        for name in [
            "NVIDIA_API_KEY",
            "TAVILY_API_KEY",
            "AGENT_LLM_MODEL",
            "AGENT_LLM_BASE_URL",
        ]:
            mp.setenv(name, "unused")
        mp.delenv("NEMO_GUARDRAILS_CONFIG_PATH", raising=False)
        mp.syspath_prepend(AGENT_DIR)
        sys.modules.pop(request.param, None)
        module = importlib.import_module(request.param)
        yield module
        sys.modules.pop(request.param, None)
    # the modules reconfigure the root logger at import time; undo that for other tests
    logging.getLogger().handlers.clear()


@pytest.fixture
def appointment_db(graph_module, tmp_path, monkeypatch):
    """Point the tools at a small template schedule and a working copy under tmp_path."""
    template = tmp_path / "template.sqlite"
    working_copy = tmp_path / "working_copy.sqlite"
    conn = sqlite3.connect(template)
    conn.execute(
        'CREATE TABLE appointment_schedule ("index" INTEGER, "datetime" TIMESTAMP, '
        '"doctor" TEXT, "appointment_type" TEXT, "patient" TEXT)'
    )
    conn.executemany(
        "INSERT INTO appointment_schedule VALUES (?, ?, ?, ?, NULL)", SLOTS
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(graph_module, "local_file_constant", str(template))
    monkeypatch.setattr(graph_module, "local_file_current", str(working_copy))
    monkeypatch.setattr(graph_module, "appointment_db_ready", False, raising=False)
    return graph_module, working_copy


def _search(module):
    return module.find_available_appointments.invoke(
        {
            "appointment_type": "Flu shots",
            "start_date": datetime.date(2025, 10, 7),
            "end_date": datetime.date(2025, 10, 13),
        }
    )


def _book(module, slot):
    return module.book_appointment.invoke(
        {
            "appointment_datetime": datetime.datetime.fromisoformat(slot),
            "appointment_type": "Flu shots",
        }
    )


def _booked_slots(working_copy):
    # a new connection only sees what was committed to the file
    conn = sqlite3.connect(working_copy)
    try:
        rows = conn.execute(
            "SELECT datetime FROM appointment_schedule WHERE patient = 'current_patient'"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def test_booking_is_saved(appointment_db):
    module, working_copy = appointment_db

    slots = _search(module)
    assert slots == [
        "2025-10-07 09:00:00",
        "2025-10-07 16:00:00",
        "2025-10-08 10:00:00",
    ]

    booked = _book(module, slots[1])
    assert list(booked["datetime"]) == ["2025-10-07 16:00:00"]
    assert _booked_slots(working_copy) == ["2025-10-07 16:00:00"]

    # same datetime as a free slot but a different appointment type: nothing more is booked
    module.book_appointment.invoke(
        {
            "appointment_datetime": datetime.datetime(2025, 10, 8, 10, 0),
            "appointment_type": "Adult physicals",
        }
    )
    assert _booked_slots(working_copy) == ["2025-10-07 16:00:00"]


def test_search_after_booking_keeps_the_booking(appointment_db):
    module, working_copy = appointment_db

    slots = _search(module)
    _book(module, slots[1])

    slots_after = _search(module)
    assert "2025-10-07 16:00:00" not in slots_after
    assert slots_after == ["2025-10-07 09:00:00", "2025-10-08 10:00:00"]
    assert _booked_slots(working_copy) == ["2025-10-07 16:00:00"]


def test_booking_before_any_search_uses_the_sample_schedule(appointment_db):
    module, working_copy = appointment_db

    _book(module, "2025-10-08 10:00:00")
    assert _booked_slots(working_copy) == ["2025-10-08 10:00:00"]


def test_booking_returns_only_the_slot_booked_now(appointment_db):
    module, working_copy = appointment_db

    # one conversation books a slot
    _book(module, "2025-10-07 16:00:00")

    # a later conversation searches and books another slot; it must not be told
    # that the earlier booking is also its own
    slots = _search(module)
    booked = _book(module, slots[0])
    assert list(booked["datetime"]) == ["2025-10-07 09:00:00"]
    assert _booked_slots(working_copy) == ["2025-10-07 09:00:00", "2025-10-07 16:00:00"]

    # booking a slot that is already taken books nothing and returns an empty table
    booked = _book(module, "2025-10-07 16:00:00")
    assert booked.empty
    assert _booked_slots(working_copy) == ["2025-10-07 09:00:00", "2025-10-07 16:00:00"]
