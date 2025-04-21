import gradio as gr
from graph import app

CHAT_HISTORY = []

def run_weather_agent(message, history):
    print("\n[Frontend Call]")

    if not message:
        return "Hi there! Ask me about the current weather or forecast in any city."

    print(f"User inputtttt: {message}")


    # Pass a plain dict so LangGraph sees your key
    final_state = app.invoke({"user_input": message, "chat_history": CHAT_HISTORY})
    print("Final state:", final_state) 
    response = final_state.get("response")
    if not response:
        response = "Sorry, I didn't catch that. Can you tell me the city and whether you're asking about the current weather or the forecast?"
    print("Response:", response)
    # get response after </think> tag
    if response:
        end = response.find("</think>")
        if end != -1:
            response = response[end + len("</think>") :].strip()
        else:
            response = "Sorry, I didn't catch that. Can you tell me the city and whether you're asking about the current weather or the forecast?"

    # Append the user message and response to the chat history
    # CHAT_HISTORY.append({"role": "user", "content": message})
    CHAT_HISTORY.append({"role": "assistant", "content": response})

    return response

gr.ChatInterface(fn=run_weather_agent, type="messages").launch()