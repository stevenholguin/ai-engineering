"""
Conversational agent - Phase 1 (plain Python, no LangChain).

This is where you see "by hand" what a framework does for you in Phase 2:
- Building the system prompt
- Defining a "tool" (native function calling) to force the model to return
  the check-in in a structured format
- Parsing the tool_calls block from the response

Requires the OPENAI_API_KEY environment variable.
"""
import os
import json
from openai import OpenAI
from models import CheckIn, VitalSigns

MODEL = "gpt-4o"  # adjust to whichever model is available on your account

SYSTEM_PROMPT = """You are CareAI, a home-monitoring assistant for patients who are
recovering or living with a chronic condition. Your role is EXCLUSIVELY to:
1. Have a brief, warm, clear conversation to collect the daily check-in
2. Ask about vital signs the patient can self-report (temperature, oxygen
   saturation if they have a pulse oximeter, heart rate, blood pressure if
   they have a monitor)
3. Ask about symptoms (pain 0-10, difficulty breathing, dizziness)
4. Ask whether they took their medication
5. At the end of the conversation, call the `register_checkin` tool with the
   data you collected

IMPORTANT RULES:
- You NEVER give diagnoses, and never interpret whether something is serious
- If the patient reports something that sounds urgent, calmly tell them this
  will be logged and a caregiver will be notified - don't over-reassure or alarm them
- If a value wasn't reported, leave it as null in the tool call, never make it up
- One question at a time, keep the conversation brief (this is for older
  adults - unhurried, but not long-winded)
"""

TOOL_REGISTER_CHECKIN = {
    "type": "function",
    "function": {
        "name": "register_checkin",
        "description": "Registers the structured patient check-in once enough information has been collected.",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature_c": {"type": ["number", "null"]},
                "spo2_pct": {"type": ["number", "null"]},
                "heart_rate_bpm": {"type": ["integer", "null"]},
                "systolic_bp": {"type": ["integer", "null"]},
                "diastolic_bp": {"type": ["integer", "null"]},
                "pain_0_10": {"type": ["integer", "null"]},
                "breathing_difficulty": {"type": ["boolean", "null"]},
                "dizziness": {"type": ["boolean", "null"]},
                "took_medication": {"type": ["boolean", "null"]},
                "notes": {"type": "string"},
            },
            "required": ["notes"],
        },
    },
}


class CareAIAgent:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.message_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, user_message: str) -> tuple[str, CheckIn | None]:
        """Sends a user message, returns (response_text, checkin_if_closed)."""
        self.message_history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self.message_history,
            tools=[TOOL_REGISTER_CHECKIN],
        )

        message = response.choices[0].message
        checkin = None

        # Keep the assistant's turn in history (needed for multi-turn context)
        assistant_entry = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        self.message_history.append(assistant_entry)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "register_checkin":
                    data = json.loads(tool_call.function.arguments)
                    checkin = self._parse_checkin(data)

        return message.content or "", checkin

    def _parse_checkin(self, data: dict) -> CheckIn:
        vitals = VitalSigns(
            temperature_c=data.get("temperature_c"),
            spo2_pct=data.get("spo2_pct"),
            heart_rate_bpm=data.get("heart_rate_bpm"),
            systolic_bp=data.get("systolic_bp"),
            diastolic_bp=data.get("diastolic_bp"),
        )
        return CheckIn(
            patient_id=self.patient_id,
            vitals=vitals,
            pain_0_10=data.get("pain_0_10"),
            breathing_difficulty=data.get("breathing_difficulty"),
            dizziness=data.get("dizziness"),
            took_medication=data.get("took_medication"),
            notes=data.get("notes", ""),
        )
