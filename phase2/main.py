"""
CareAI - Phase 2: home check-in CLI (LangChain, RAG, Pydantic).

Usage:
    export OPENAI_API_KEY="your-api-key"
    python main.py

First run will build the local Chroma vectorstore from knowledge/ - this
can take a few seconds.
"""
from agent import CareAIAgent
from rules import detect_alerts
from storage import init_db, save_checkin, patient_history


def main():
    init_db()
    patient_id = input("Patient ID: ").strip() or "demo_patient"
    agent = CareAIAgent(patient_id)

    print("\n--- CareAI (Phase 2): daily check-in ---")
    print("(type 'exit' at any time to stop - try asking a question like")
    print(" 'what does it mean if my blood pressure is 185?')\n")

    message = "Hi"
    while True:
        reply, checkin = agent.send(message)
        print(f"\nCareAI: {reply}\n")

        if checkin is not None:
            alerts = detect_alerts(checkin)
            save_checkin(checkin, alerts)

            if alerts:
                print("🚨 ALERT DETECTED - caregiver would be notified:")
                for a in alerts:
                    print(f"   - {a}")
            else:
                print("✅ Check-in recorded, no alerts.")
            break

        message = input("You: ").strip()
        if message.lower() in ("exit", "quit"):
            print("Check-in interrupted, nothing saved.")
            break

    print(f"\n--- History for {patient_id} ---")
    for h in patient_history(patient_id):
        print(h)


if __name__ == "__main__":
    main()
