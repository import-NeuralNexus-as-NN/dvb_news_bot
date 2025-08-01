import json
from pathlib import Path

print("[SUMMARIZER MODULE] summarizer.py загружен")

with open("config/prompts.json", "r", encoding="utf-8-sig") as f:
    PROMPTS = json.load(f)

with open("config/llm_config.json", "r", encoding="utf-8-sig") as f:
    llm_conf = json.load(f)

llm_backend = llm_conf.get("llm_backend", 2)

if llm_backend == 0:
    print("Используется OpenAI")
    from openai import OpenAI
    client = OpenAI(
        base_url=llm_conf["base_url_Open_AI"],
        api_key=llm_conf["api_key_Open_AI"],
    )
elif llm_backend == 1:
    print("Используется LangChain+Groq")
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    client = ChatGroq(
        model=llm_conf.get("model_LANGCHAIN", "llama3-8b-8192"),
        api_key=llm_conf["api_key_LANGCHAIN"]
    )
else:
    print("Используется LangChain+Mistral")
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import HumanMessage

    client = ChatMistralAI(
        model_name=llm_conf.get("model_LANGCHAIN_mistral", "mistral-small"),
        api_key=llm_conf["api_key_LANGCHAIN_mistral"]
    )

def summarize_text(text: str, department: str = "") -> str:
    if not department:
        return ""
    prompt = PROMPTS["summary"].format(department=department, text=text)
    print("[SUMMARIZER] Prompt сформирован")
    try:
        if llm_backend == 0:
            response = client.chat.completions.create(
                model=llm_conf["model_Open_AI"],
                extra_body={},
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content.strip()
        else:
            summary = client.invoke([HumanMessage(content=prompt)]).content.strip()

        print(f"[SUMMARIZER] Готовое резюме: {summary}")
        return summary
    except Exception as e:
        print(f"[SUMMARIZER ERROR] {e}")
        return ""