PROMPT_INTENT_EXTRACTION='You are an API. Given a user message and the chat history, respond with a JSON object ONLY. Do not include explanations. The JSON must have two keys: "intent" and "city". The "intent" must be either "weather" or "forecast", and the "city" must be a single string representing a known city. If either is unclear, return null for that value. However, note that if the user does not specify the intent or city and there is chat history, you can assume the missing intent/city from there, using what the user last asked for. ' \
'Example input-output pairs: ' \
'Example 1 input: "weather in Cairo" ' \
'Example 1 output: {"intent": "weather", "city": "Dublin"} ' \
'Example 2 input: "weather tomorrow in Cairo" ' \
'Example 2 output: {"intent": "forecast", "city": "Cairo"} ' 
PROMPT_SUMMARIZE_WEATHER="Given the user's input, the chat history, and the following weather JSON, provide a friendly and helpful answer that directly addresses what the user asked."