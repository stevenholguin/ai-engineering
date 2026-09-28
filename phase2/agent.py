"""
Conversational agent - Phase 2 (LangChain).

What changes vs. Phase 1 (plain Python + OpenAI SDK):
- Structured extraction uses a Pydantic `args_schema` instead of a raw JSON
  schema dict - LangChain generates the schema for us.
- The agent now has TWO tools: `register_checkin` (same purpose as before)
  and `search_care_protocols` (new - RAG).
- Because there are now two tools that can both be called, and one of them
  (search) needs its result fed back to the model before it can respond,
  this introduces a proper tool-calling LOOP: call the model -> if it wants
  a tool, run it -> feed the result back -> call the model again -> repeat
  until it either answers in plain text or calls register_checkin.
  This loop is exactly the kind of thing LangGraph (Phase 3) will let us
  model explicitly as a graph instead of a while-loop.
- Conversation memory is still a plain list of LangChain message objects
  here (BaseMessage subclasses) rather than LangChain's higher-level memory
  classes - intentionally, so the mechanics stay visible one more phase
  before we hand persistence off to LangGraph's checkpointing in Phase 3.
"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from models import CheckIn, VitalSigns
from rag import search_care_protocols

MODEL = "gpt-4o"
MAX_TOOL_ITERATIONS = 5  # safety cap so a tool-call loop can never run forever

SYSTEM_PROMPT = """You are CareAI, a home-monitoring assistant for patients who are
recovering or living with a chronic condition. Your role is:
1. Have a brief, warm, clear conversation to collect the daily check-in
2. Ask about vital signs the patient can self-report (temperature, oxygen
   saturation if they have a pulse oximeter, heart rate, blood pressure if
   they have a monitor)
3. Ask about symptoms (pain 0-10, difficulty breathing, dizziness)
4. Ask whether they took their medication
5. If the patient asks a question about what a symptom or reading might
   mean, use the `search_care_protocols` tool to ground your answer in the
   care protocol documents - always mention that this is general guidance,
   not a diagnosis, and that a caregiver will review the check-in
6. Once you have collected enough information, call the `register_checkin`
   tool with the data

IMPORTANT RULES:
- You NEVER give diagnoses, and never interpret whether something is serious
- If the patient reports something that sounds urgent, calmly tell them this
  will be logged and a caregiver will be notified - don't over-reassure or alarm them
- If a value wasn't reported, leave it as null when calling register_checkin
- One question at a time, keep the conversation brief
"""


class CheckInArgs(BaseModel):
    """Arguments schema for the register_checkin tool - this replaces the
    hand-written JSON schema dict from Phase 1."""
    temperature_c: Optional[float] = Field(default=None, description="Body temperature in Celsius")
    spo2_pct: Optional[float] = Field(default=None, description="Oxygen saturation percentage")
    heart_rate_bpm: Optional[int] = Field(default=None, description="Heart rate in beats per minute")
    systolic_bp: Optional[int] = Field(default=None, description="Systolic blood pressure")
    diastolic_bp: Optional[int] = Field(default=None, description="Diastolic blood pressure")
    pain_0_10: Optional[int] = Field(default=None, description="Self-reported pain level, 0-10")
    breathing_difficulty: Optional[bool] = Field(default=None)
    dizziness: Optional[bool] = Field(default=None)
    took_medication: Optional[bool] = Field(default=None)
    notes: str = Field(description="Free-text summary of what the patient reported")


@tool("register_checkin", args_schema=CheckInArgs)
def register_checkin_tool(**kwargs) -> str:
    """Registers the structured patient check-in once enough information has
    been collected. Call this once, at the end of the conversation."""
    # Intentionally a no-op: the agent intercepts the raw tool_call args
    # itself (see CareAIAgent.send) instead of letting this body run, so the
    # check-in data is available to the CLI/rules/storage layer directly.
    return "Check-in captured."


@tool("search_care_protocols")
def search_care_protocols_tool(query: str) -> str:
    """Search the home-care protocol documents for guidance relevant to a
    patient's question or reported symptom (e.g. what a given fever or
    blood pressure reading might mean, or medication adherence guidance).
    Always present results as general guidance, never as a diagnosis."""
    return search_care_protocols(query)


class CareAIAgent:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        llm = ChatOpenAI(model=MODEL, api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm_with_tools = llm.bind_tools(
            [register_checkin_tool, search_care_protocols_tool]
        )
        self.system_message = SystemMessage(content=SYSTEM_PROMPT)
        self.history: list = []  # HumanMessage / AIMessage / ToolMessage

    def send(self, user_message: str) -> tuple[str, Optional[CheckIn]]:
        """Sends a user message, returns (response_text, checkin_if_closed).
        Internally may call the model multiple times if it uses the RAG
        tool before producing a final answer or the check-in tool call."""
        self.history.append(HumanMessage(content=user_message))

        for _ in range(MAX_TOOL_ITERATIONS):
            ai_message = self.llm_with_tools.invoke(
                [self.system_message] + self.history
            )
            self.history.append(ai_message)

            if not ai_message.tool_calls:
                return ai_message.content, None

            checkin = None
            tool_results = []

            for call in ai_message.tool_calls:
                if call["name"] == "register_checkin":
                    checkin = self._parse_checkin(call["args"])
                elif call["name"] == "search_care_protocols":
                    result = search_care_protocols_tool.invoke(call["args"])
                    tool_results.append(
                        ToolMessage(content=result, tool_call_id=call["id"])
                    )

            if checkin is not None:
                return ai_message.content or "Thanks, your check-in has been recorded.", checkin

            # Only search-tool calls happened: feed results back and loop
            # so the model can produce its final answer grounded in them.
            self.history.extend(tool_results)

        return "Sorry, something went wrong processing that - let's try again.", None

    def _parse_checkin(self, args: dict) -> CheckIn:
        vitals = VitalSigns(
            temperature_c=args.get("temperature_c"),
            spo2_pct=args.get("spo2_pct"),
            heart_rate_bpm=args.get("heart_rate_bpm"),
            systolic_bp=args.get("systolic_bp"),
            diastolic_bp=args.get("diastolic_bp"),
        )
        return CheckIn(
            patient_id=self.patient_id,
            vitals=vitals,
            pain_0_10=args.get("pain_0_10"),
            breathing_difficulty=args.get("breathing_difficulty"),
            dizziness=args.get("dizziness"),
            took_medication=args.get("took_medication"),
            notes=args.get("notes", ""),
        )
