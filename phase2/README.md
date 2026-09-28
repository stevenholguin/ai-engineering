# CareAI — Fase 2: LangChain

Misma idea que la Fase 1 (agente de monitoreo domiciliario), reconstruida
con LangChain. El código sigue en inglés; esta explicación en español.

## Qué se agregó respecto a la Fase 1

| Concepto | Fase 1 (Python puro) | Fase 2 (LangChain) |
|---|---|---|
| Extracción estructurada | JSON schema escrito a mano | `args_schema` con Pydantic (`CheckInArgs`) |
| Modelos de datos | `dataclasses` | Pydantic (`BaseModel`) |
| Herramientas | 1 tool (`register_checkin`) | 2 tools (`register_checkin` + `search_care_protocols`) |
| Conocimiento externo | Ninguno | RAG sobre `knowledge/*.md` con Chroma |
| Flujo de llamadas | Una sola llamada al modelo por turno | Loop de tool-calling (puede llamar RAG, recibir el resultado, y responder) |

## Cómo correrlo

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="tu-api-key"
python main.py
```

En la primera ejecución se construye el vectorstore local (Chroma) a partir
de los documentos en `knowledge/` — tarda unos segundos. Prueba preguntando
algo como *"what does it mean if my blood pressure is 185?"* durante el
check-in para ver el RAG en acción.

## Estructura

```
cuidaia-fase2/
├── main.py                    # CLI
├── agent.py                    # agente LangChain: tools, loop, memoria
├── models.py                    # CheckIn y VitalSigns (ahora Pydantic)
├── rag.py                        # construcción y consulta del vectorstore
├── rules.py                       # motor de reglas de alerta (sin cambios)
├── storage.py                      # persistencia SQLite (sin cambios)
├── knowledge/                       # documentos fuente para el RAG
│   ├── fever_protocol.md
│   ├── blood_pressure_protocol.md
│   ├── pain_management.md
│   └── general_home_care.md
└── requirements.txt
```

## El concepto más importante de esta fase: el loop de tool-calling

En la Fase 1 el modelo hacía **una sola llamada** por turno: o respondía
texto, o llamaba `register_checkin`. Ahora, con dos herramientas, puede
pasar esto:

1. El paciente pregunta "¿qué significa una presión de 185?"
2. El modelo llama `search_care_protocols` (no responde texto todavía)
3. Tu código ejecuta la búsqueda y le devuelve el resultado como `ToolMessage`
4. **El modelo se vuelve a llamar**, ahora con ese resultado en el historial,
   y ahí sí genera la respuesta final en texto

Por eso `agent.py` ya no es una sola llamada — es un `for` con un límite de
iteraciones (`MAX_TOOL_ITERATIONS`) que repite "llamar al modelo → ejecutar
tools → repetir" hasta que el modelo responde en texto plano o llama
`register_checkin`.

Esto es exactamente el problema que LangGraph (Fase 3) resuelve de forma
más explícita: en vez de un `while`/`for` escondiendo la lógica de
decisión, vas a modelar este mismo flujo como un grafo con nodos y bordes
condicionales — y vas a poder ver visualmente por qué caminos pasó cada
conversación.

## Por qué `register_checkin` no ejecuta nada en su propio cuerpo

Fíjate que la función decorada con `@tool("register_checkin", ...)` solo
devuelve un string fijo y no hace nada más. Esto es intencional: como el
check-in necesita terminar la conversación y disparar `rules.py` +
`storage.py` (que viven fuera del agente), es más claro interceptar los
argumentos crudos del `tool_call` en `CareAIAgent.send()` en vez de dejar
que LangChain ejecute la función y usar su valor de retorno. La tool
"real" para LangChain solo existe para darle al modelo el schema correcto.

## Limitaciones conocidas (que resolveremos en fases siguientes)

- El loop de tool-calling vive en un `for` con límite fijo — no hay control
  explícito de qué rama tomar según el nivel de riesgo detectado →
  **Fase 3 (LangGraph)**
- La memoria sigue siendo una lista en RAM: si reinicias el proceso, se
  pierde el historial de la conversación en curso (aunque los check-ins ya
  cerrados sí persisten en SQLite) → **Fase 3 (LangGraph con checkpointing)**
- Las tools siguen viviendo dentro del mismo proceso Python →
  **Fase 3.5 (MCP)**

## Siguiente paso

Fase 3: reconstruir esto con LangGraph — el flujo como grafo de estados
explícito, con ramas de decisión según el nivel de riesgo y checkpointing
para que el check-in sobreviva entre sesiones.
