# The system: which services exist, and who calls whom? (Draw it as arrows, e.g. frontend -> checkout.)
# Normal behaviour: what are the normal CPU, memory and latency levels per service? Add a little random noise, because real data is never flat.
# Faults: for each reason (high CPU, high memory, network delay), what changes in the data? Which metric goes up, and on which service?
# Propagation: when a service is slow, what happens to its callers, and how many minutes later?
# Traps: what will make the agent’s job hard? At least two. (Hint: one is about symptoms, one is about noise.)
# Ground truth: for each case, what exactly do you store as the answer? It must use the same format as your output in question 3.
# How many cases, and how do you make sure every component and every reason appears?

import json
import os
import re
import sys

import pandas as pd
import numpy as np

from openai import OpenAI

# 1. model (Ollama)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  
MODEL = "qwen2.5:7b"

# 2. Tools: plain Python functions that read the data of the current case
DATA_DIR = "data"
DF = None  # metrics of the current case, set by load_case()

# load the data of the current case into the global DF variable
def load_case(case_id):
    global DF
    DF = pd.read_csv(os.path.join(DATA_DIR, f"{case_id}.csv"))

# list all components and their callers in natural language 
def list_components():
    with open(os.path.join(DATA_DIR, "system.json")) as f:
        calls = json.load(f)
    return "; ".join(f"{s} calls {', '.join(c) if c else 'nothing'}" for s, c in calls.items())

def kpi_change(component, kpi):
    """Biggest jump of one KPI during the hour, and the average before and after it.
    kpi is one of "cpu", "memory" or "latency"."""
    if kpi not in ("cpu", "memory", "latency"):
        return "kpi must be cpu, memory or latency"
    s = DF[DF["service"] == component]
    if s.empty:
        names = ", ".join(sorted(DF["service"].unique()))
        return f"'{component}' is not a component. Components are: {names}. KPIs are cpu, memory, latency."
    v, times = s[kpi].to_numpy(), s["time"].to_numpy()
    jumps = np.abs(np.diff(v))                 # change from one minute to the next
    jumps[:4] = jumps[-4:] = 0                 # keep a few minutes on both sides to compare
    k = int(np.argmax(jumps)) + 1              # minute with the biggest jump
    change = (v[k:].mean() - v[:k].mean()) / v[:k].mean() * 100
    return (f"{kpi}: biggest jump at {times[k]}, from {v[k-1]:.0f} to {v[k]:.0f}; "
            f"average before {v[:k].mean():.0f}, average after {v[k:].mean():.0f} ({change:+.0f}%)")

TOOLS = {"list_components": list_components, "kpi_change": kpi_change}


# 3. Prompt
SYSTEM = """You find the root cause of one failure in a small microservice system.
Possible reasons: CPU fault, memory fault, network delay.

Tools:
- list_components(): all services and which services each one calls
- kpi_change(component, kpi): the biggest jump of one KPI during the hour, with the average before and after it; kpi is cpu, memory or latency

How to diagnose (in this order):
1. Look at the system: call list_components to see which service calls which.
2. Find changes: check cpu, memory and latency of EVERY service.
3. Keep only real faults: a real fault changes the average a lot (more than 50%) and stays.
   A change of a few percent, or a short spike that goes back to normal, is noise. Ignore it.
4. Find the root cause among the real faults:
   - Usually, when a service is slow, the services that call it become slow too, a little later.
     So the root cause is usually the faulty service furthest down the call chain, and its fault starts first.
   - The reason comes from the KPI of that fault: cpu -> CPU fault, memory -> memory fault, only latency -> network delay.
   - The time is the first minute of that fault.

Reply with ONE JSON object each turn:
{"thought": "...", "action": "<tool name>", "args": {...}}
When you know the answer:
{"thought": "...", "action": "final_answer", "args": {"component": "...", "reason": "...", "time": "HH:MM"}}"""

QUESTION = ("Between 00:00 and 01:00 one failure happened. "
            "Find the root cause component, the reason and the time it started.")

def run(question, max_steps=20):
    # 4. memory = the message list
    messages = [{"role": "system", "content": SYSTEM},  
                {"role": "user", "content": question}]
    
    # 5. loop with a step limit
    reviewed = False
    for step in range(max_steps):  
        # if last step, tell the model to reply with final_answer
        if step == max_steps - 1:
            messages.append({"role": "user", "content": "Last step: reply with final_answer now."})
        
        # call the model, get the reply, temperature=0 for highest probability output 
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
            # ask model to review the data first, if it hasn't done so yet
            if not reviewed and step < max_steps - 1:
                reviewed = True
                with open(os.path.join(DATA_DIR, "system.json")) as f:
                    callees = json.load(f).get(args.get("component"), [])
                messages.append({"role": "user", "content":
                    f"Before you answer: {args.get('component')} calls {', '.join(callees) or 'nothing'}. "
                    "Did you check cpu, memory and latency of those services? If one of them has a real fault "
                    "that starts earlier, it is the root cause. Check them if needed, then give final_answer again."})
                continue
            return args
        # case name not in TOOLS -> tell the model, case name in TOOLS -> call the function and return the observation
        if name not in TOOLS:
            obs = f"Unknown tool {name}. Use one of {list(TOOLS)}."
        else:
            try:
                obs = TOOLS[name](**args)
            except Exception as e:                        # bad arguments -> tell the model
                obs = f"Tool error: {e}"
        
        print(f"Observation: {obs}")
        # 9. append the observation to the memory
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    return {"error": "no answer within the step limit"}

if __name__ == "__main__":
    load_case("case_01")             
    print(run(QUESTION))