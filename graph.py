import os
import json
import requests
import re
from dotenv import load_dotenv
from prompts import PROMPT_INTENT_EXTRACTION, PROMPT_SUMMARIZE_WEATHER
from datetime import datetime, timedelta
import pytz

from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOllama


# Load .env
load_dotenv()

# Set up local DeepSeek agent
llm = ChatOllama(model="deepseek-r1:8b", temperature=0.0)

# Get API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
# print("OpenWeather API Key:", OPENWEATHER_API_KEY)

# Custom state
# class GraphState(dict):
#     pass

EGYPT_TZ = pytz.timezone("Africa/Cairo")

class GraphState(dict):
    def __init__(self, *args, **kwargs):
        print("Initializing GraphState with:", args, kwargs)
        super().__init__(*args, **kwargs)
        if "chat_history" not in self:
            self["chat_history"] = []



### NODE 1: Extract Intent & City ###
def extract_intent_city(state):
    user_input = state.get("user_input")
    print(state)
    # user_input = state["user_input"]
    print("\n[Node: Extract Intent & City]")
    print(f"User input: {user_input}")

    prompt = PROMPT_INTENT_EXTRACTION
    if not prompt:
        raise ValueError("Prompt for intent extraction is not set in the environment variables.")
    print("\nPrompt 1:", prompt)

    # Build full message list
    raw_messages = [
        {"role": "system", "content": prompt},
    ]
    
    # Add chat history if available
    if "chat_history" in state:
        raw_messages.extend(state["chat_history"])
    
    # Add user input
    raw_messages.append({"role": "user", "content": user_input})

    # Convert to LangChain message objects
    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage
    }

    langchain_messages = [role_map[msg["role"]](content=msg["content"]) for msg in raw_messages]

    print("\n\n######LangChain Messages:", langchain_messages)
    print("######\n\n")

    response = llm.invoke(langchain_messages)

    print("Raw LLM Response:", response.content)

    city = None
    intent = None

    try:
        # Extract JSON from the response
        match = re.search(r"\{.*?\}", response.content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        json_str = match.group(0)
        print("Extracted JSON String:", json_str)
        response_json = json.loads(json_str)

        print("Parsed JSON:", response_json)
        
        # extract city and intent
        city = response_json.get("city")
        intent = response_json.get("intent")
        print("Extracted City:", city)
        print("Extracted Intent:", intent)
        # Check if the response is valid JSON
        if not city or not intent:
            raise ValueError("City or intent not found in the response.")
        
    except Exception as e:
        print(f"[Intent Extraction Error] {e}")
        city = None
        intent = None

    state["city"] = city
    state["intent"] = intent
    
    if "chat_history" not in state:
        state["chat_history"] = []

    state["chat_history"].append({"role": "user", "content": user_input})
    # remove think tags and text between them from ai responsne before appending it
    if "<think>" in response.content:
        start = response.content.find("<think>")
        end = response.content.find("</think>")
        if end != -1:
            response.content = response.content[:start] + response.content[end + len("</think>"):]
        else:
            response.content = response.content[start + len("<think>"):]

    print("Response content stripped:", response.content)

    state["chat_history"].append({"role": "assistant", "content": response.content})

    return state


### NODE 2: Fallback if Missing Info ###
def fallback_node(state):
    print("\n[Node: Fallback]")
    print("Missing intent or city. Asking user again.")
    state["response"] = "Sorry, I didn't catch that. Can you tell me the city and whether you're asking about the current weather or the forecast?"
    return state


### ROUTER NODE ###
def route_node(state):
    print("\n[Node: Router]")
    print(f"Routing based on intent: {state.get('intent')}")
    if state.get("intent") == "forecast":
        return "forecast_node"
    elif state.get("intent") == "weather":
        return "weather_node"


### NODE 3: Current Weather ###
def get_current_weather(state):
    print("\n[Node: Current Weather]")
    city = state.get("city")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url)
    data = r.json()
    print("Weather API Response:", json.dumps(data, indent=2))
    state["weather_data"] = data
    return state


### NODE 4: Forecast Weather (Every 12 Hours for 3 Days) ###
def get_forecast_weather(state):
    print("\n[Node: Forecast Weather]")
    city = state.get("city")
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url)
    data = r.json()

    forecast_list = data.get("list", [])
    filtered = []

    # Define tomorrow's date in Egypt time
    now = datetime.now(EGYPT_TZ)
    tomorrow = now + timedelta(days=1)
    tomorrow_date = tomorrow.date()

    # Loop through all forecast entries
    for entry in forecast_list:
        # Convert timestamp to Egypt local time
        utc_time = datetime.utcfromtimestamp(entry["dt"]).replace(tzinfo=pytz.utc)
        local_time = utc_time.astimezone(EGYPT_TZ)

        # Check if the date is exactly tomorrow
        if local_time.date() == tomorrow_date:
            # Match hours close to 00:00 or 12:00
            if local_time.hour in [23, 0, 1]:
                entry["local_time_label"] = "00:00"
                filtered.append(entry)
            elif local_time.hour in [11, 12, 13]:
                entry["local_time_label"] = "12:00"
                filtered.append(entry)

        # Optional: simplify the dt_txt field to just the date
        if "dt_txt" in entry:
            entry["dt_txt"] = local_time.strftime("%Y-%m-%d")

        # remove "temp_min" and "temp_max" from the entry
        if "main" in entry:
            entry["main"].pop("temp_min", None)
            entry["main"].pop("temp_max", None)

        # Stop when we have both entries (max 2)
        if len(filtered) >= 2:
            break

    data["list"] = filtered
    print("Filtered Forecast Data:", json.dumps(filtered, indent=2, default=str))
    state["weather_data"] = data
    return state


def summarize_weather(state):
    print("\n[Node: LLM Summarize]")
    prompt = PROMPT_SUMMARIZE_WEATHER
    user_input = state.get("user_input", "")
    weather_json = json.dumps(state.get("weather_data"), indent=2)

    # Build full message list
    raw_messages = [
        {"role": "system", "content": prompt},
        *state.get("chat_history", []),
        {"role": "user", "content": f"My latest input was: {user_input}."},
        {"role": "assistant", "content": f"Here is the weather data: {weather_json}"}
    ]

    # Convert to LangChain message objects
    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage
    }

    # Convert to HumanMessage format
    # langchain_messages = [HumanMessage(content=msg["content"]) for msg in messages if msg["role"] == "user"]
    langchain_messages = [role_map[msg["role"]](content=msg["content"]) for msg in raw_messages]

    if state.get("intent") == "forecast":
        from datetime import datetime, timedelta
        today = datetime.now()
        tomorrow = today + timedelta(days=1)

        # Add date to the prompt
        langchain_messages.append(SystemMessage(content=f"Tomorrow's date is: {tomorrow.strftime('%Y-%m-%d')}."))
        
    print("\n\n######LangChain Messages2:", langchain_messages)
    print("######\n\n")

    response = llm.invoke(langchain_messages)
    print("Summary Response:", response.content)

    # state["chat_history"].append({"role": "assistant", "content": response.content})
    state["response"] = response.content
    return state


# Graph Definition
workflow = Graph()

# Add nodes
workflow.add_node("extract_intent", extract_intent_city)
workflow.add_node("fallback", fallback_node)
workflow.add_node("weather_node", get_current_weather)
workflow.add_node("forecast_node", get_forecast_weather)
workflow.add_node("summarize", summarize_weather)

# Set entry point
workflow.set_entry_point("extract_intent")

# Route after extract_intent: fallback if data missing, else decide based on intent
def route_intent(state):
    if not state.get("intent") or not state.get("city"):
        return "fallback"
    elif state.get("intent") == "forecast":
        return "forecast_node"
    else:
        return "weather_node"

workflow.add_conditional_edges(
    "extract_intent",
    route_intent,
    {
        "fallback": "fallback",
        "forecast_node": "forecast_node",
        "weather_node": "weather_node"
    }
)

# After API call, go to summarization
workflow.add_edge("forecast_node", "summarize")
workflow.add_edge("weather_node", "summarize")

# End points
workflow.add_edge("fallback", END)
# workflow.add_edge("summarize", END)

workflow.set_finish_point("summarize")

# Compile
app = workflow.compile()