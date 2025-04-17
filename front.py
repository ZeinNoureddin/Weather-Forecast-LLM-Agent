import gradio as gr
from graph import app

def run_weather_agent(message, history):
    print("\n[Frontend Call]")

    if not message:
        return "Hi there! Ask me about the current weather or forecast in any city."

    print(f"User inputtttt: {message}")
    # Pass a plain dict so LangGraph sees your key
    final_state = app.invoke({"user_input": message})
    return final_state.get("response", "Something went wrong.")

# def run_weather_agent(message, history):
#     print("\n[Frontend Call]")

#     if not message:   
#         return "Hi there! Ask me about the current weather or forecast in any city."

#     print(f"User inputtttt: {message}")
#     final_state = app.invoke({"user_input": message})
    
#     print("Final state:", final_state)

#     response = final_state.get("response", "Something went wrong.")
#     print("Returning response:", response)
#     return response


gr.ChatInterface(fn=run_weather_agent, type="messages").launch()
