import os
import json
import requests
from dotenv import load_dotenv

from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage
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

class GraphState(dict):
    def __init__(self, *args, **kwargs):
        print("Initializing GraphState with:", args, kwargs)
        super().__init__(*args, **kwargs)


### NODE 1: Extract Intent & City ###
def extract_intent_city(state):
    user_input = state.get("user_input")
    print(state)
    # user_input = state["user_input"]
    print("\n[Node: Extract Intent & City]")
    print(f"User input: {user_input}")

    prompt = os.getenv("PROMPT_INTENT_EXTRACTION")
    if not prompt:
        raise ValueError("Prompt for intent extraction is not set in the environment variables.")
    print("Prompt:", prompt)
    response = llm.invoke([HumanMessage(content=f"{prompt}\n{user_input}")])
    print("Raw LLM Response:", response.content)

    city = None
    intent = None

    try:
        # find json in the response
        start = response.content.find("{")
        end = response.content.rfind("}") + 1
        json_str = response.content[start:end]
        print("Extracted JSON String:", json_str)
        # parse the json
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
        
    except:
        pass

    state["city"] = city
    state["intent"] = intent
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
    
    # Keep only every 4th item (approx 12h interval), up to 6 entries (3 days)
    filtered = data.get("list", [])[::4][:6]
    data["list"] = filtered

    print("Filtered Forecast Data:", json.dumps(filtered, indent=2))
    state["weather_data"] = data
    return state


### NODE 5: Summarize ###
def summarize_weather(state):
    print("\n[Node: LLM Summarize]")
    prompt = os.getenv("PROMPT_SUMMARIZE_WEATHER")
    weather_json = json.dumps(state.get("weather_data"), indent=2)
    response = llm.invoke([HumanMessage(content=f"{prompt}\n{weather_json}")])
    print("Summary Response:", response.content)
    # state["response"] = response.content
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
