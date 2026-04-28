with open('python/app/agents/chat_router.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_sig = 'async def route_request(self, user_query: str, available_datasets: list[str]) -> ForecastRequest:'
new_sig = 'async def route_request(self, user_query: str, available_datasets: dict[str, str]) -> ForecastRequest:'

old_avail = 'f"AVAILABLE DATASETS IN SYSTEM: {\', \'.join(available_datasets)}\\n\\n"'
new_avail = 'f"AVAILABLE DATASETS IN SYSTEM:\\n{dataset_info}\\n\\n"'

old_log = 'logger.info("ChatRouterAgent: Translating natural language for prompt=\'%s\'", user_query)'
new_log = 'logger.info("ChatRouterAgent: Translating natural language for prompt=\'%s\'", user_query)\n        \n        dataset_info = "\\n".join([f"- {name}: {desc}" for name, desc in available_datasets.items()])'

text = text.replace(old_sig, new_sig)
text = text.replace(old_avail, new_avail)
text = text.replace(old_log, new_log)

with open('python/app/agents/chat_router.py', 'w', encoding='utf-8') as f:
    f.write(text)
