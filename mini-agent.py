# The system: which services exist, and who calls whom? (Draw it as arrows, e.g. frontend -> checkout.)
# Normal behaviour: what are the normal CPU, memory and latency levels per service? Add a little random noise, because real data is never flat.
# Faults: for each reason (high CPU, high memory, network delay), what changes in the data? Which metric goes up, and on which service?
# Propagation: when a service is slow, what happens to its callers, and how many minutes later?
# Traps: what will make the agent’s job hard? At least two. (Hint: one is about symptoms, one is about noise.)
# Ground truth: for each case, what exactly do you store as the answer? It must use the same format as your output in question 3.
# How many cases, and how do you make sure every component and every reason appears?

import json, re
from openai import OpenAI

# 1. model (Ollama)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  
MODEL = "qwen2.5:3b"

# 2. Tools: plain Python functions (toy data; a real version reads the metric CSV)
def list_components():
    return ["docker_001", "docker_002", "db_003"]

def kpi_change(component, kpi):
    data = {("docker_002", "cpu"): "normal 20%, peak 80% in the window, rising from 00:10"}
    return data.get((component, kpi), "no change")

TOOLS = {"list_components": list_components, "kpi_change": kpi_change}

# 3. Prompt
SYSTEM = """You find the root cause of a failure.
Tools:
- list_components(): names of all components
- kpi_change(component, kpi): how a KPI changed in the failure window; kpi is cpu, memory or network
Reply with ONE JSON object each turn:
{"thought": "...", "action": "<tool name>", "args": {...}}
When you know the answer:
{"thought": "...", "action": "final_answer", "args": {"component": "...", "reason": "..."}}"""

def run(question, max_steps=8):
    # 4. memory = the message list
    messages = [{"role": "system", "content": SYSTEM},  
                {"role": "user", "content": question}]
    
    # 5. loop with a step limit
    for step in range(max_steps):  
        # call the model, get the reply, temperature=0 for highest probaiility output                   
        reply = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0).choices[0].message.content
        # append the reply to the memory
        messages.append({"role": "assistant", "content": reply})
        # 6. log every step
        print(f"--- step {step}\n{reply}")                
        # 7. parse and check the reply: must be a JSON object with "action" and optional "args"
        try:                                              
            act = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
            # name is the tool name or "final_answer", args is a dict with the arguments for the tool
            name, args = act["action"], act.get("args", {})
        except Exception:
            messages.append({"role": "user", "content": "Invalid format. Reply with ONE JSON object."})
            continue
        
        # 8. execute the action, name is either a tool or "final_answer"
        if name == "final_answer":
            return args
        # case name not in TOOLS -> tell the model, case name in TOOLS -> call the function and return the observation
        if name not in TOOLS:
            obs = f"Unknown tool {name}. Use one of {list(TOOLS)}."
        else:
            try:
                obs = TOOLS[name](**args)
            except Exception as e:                        # bad arguments -> tell the model
                obs = f"Tool error: {e}"
        
        # 9. append the observation to the memory
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    return {"error": "no answer within the step limit"}

if __name__ == "__main__":
    print(run("Between 00:00 and 00:30 one failure happened. Which component, and why?"))