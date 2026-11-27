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
CODER_MODEL = "qwen2.5-coder:7b"  # LLM 2: writes pandas code for query_data

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


CODER_SYSTEM = """You write Python (pandas) code to answer a question about monitoring data.
A DataFrame named DF is already loaded. Its columns are exactly: "time", "service", "cpu", "memory", "latency".
- time: "HH:MM" string, one row per service per minute from 00:00 to 00:59 (sorted by time)
- service: frontend, checkout, payment, catalog, db
- cpu and memory are percentages, latency is in milliseconds; all have some random noise
First rows of DF:
{head}
pandas is imported as pd and numpy as np.
Rules:
- Put the answer in a variable named result (a number, string, Series or small DataFrame of at most 20 rows).
- For times, use the values of the "time" column, never the row index.
- Find when a change starts from the data, separately for each service; never split the hour at a fixed minute.
- Do not print, plot, read or write files.
- Rows of different services are mixed. Always group by service first (DF.groupby("service")) before diff, shift or mean.
- Never return the raw rows. Return a summary: at most one row per service and KPI.
- When asked whether or when a KPI changes, compute for each service and KPI:
  the minute with the biggest jump (diff within the service), the average before and after that minute,
  and the change in percent. Report service, kpi, start_time, average_before, average_after, change_percent.
  A short spike shows a big jump but almost no change in the averages.
Reply with only one Python code block."""


def query_data(question):
    """LLM 2 writes pandas code for the question; the code runs on DF; the agent gets the result table."""
    system = CODER_SYSTEM.replace("{head}", DF.head(5).to_string(index=False))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    error = ""
    for attempt in range(2):                              # one retry if the code fails
        reply = client.chat.completions.create(
            model=CODER_MODEL, messages=messages, temperature=0).choices[0].message.content
        m = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
        code = m.group(1) if m else reply
        print(f"    [coder] code:\n{code.strip()}")
        try:
            env = {"DF": DF.copy(), "pd": pd, "np": np}
            exec(code, env)                               # run the code
            result = env["result"]
            if hasattr(result, "__len__") and not isinstance(result, str) and len(result) > 20:
                raise ValueError(f"result has {len(result)} rows; return a summary with at most 20 rows")    
            text = result.to_string() if hasattr(result, "to_string") else str(result)
            return f"Result of the code for '{question}':\n{text[:2000]}"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            print(f"    [coder] error: {error}")
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"The code failed with: {error}\n"
                          f"The columns of DF are exactly {list(DF.columns)}. Fix the code and reply with the code only."}]
    return f"The helper could not answer (code error: {error}). Ask in a simpler way."

TOOLS = {"list_components": list_components, "query_data": query_data}


# 3. Prompt
SYSTEM = """You find the root cause of one failure in a small microservice system.
Possible reasons: CPU fault, memory fault, network delay.

Tools:
- list_components(): all services and which services each one calls
- query_data(question): ask any question about the cpu, memory and latency data, in plain English.
  A helper writes and runs pandas code and returns the result table. Ask one clear question at a time.

How to diagnose (in this order):
1. Look at the system: call list_components to see which service calls which.
2. Find changes: use query_data to check cpu, memory and latency of EVERY service (when each one changes, and its average before and after the change).
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
            start = reply.index("{")                                  # where the JSON begins
            act, _ = json.JSONDecoder().raw_decode(reply[start:])     # stops at the end of the first object
            # name is the tool name or "final_answer", args is a dict with the arguments for the tool
            name, args = act["action"], act.get("args", {})
        except Exception as e:
            note = f"Invalid format ({e}). Reply with ONE JSON object only."
            print(f"Note: {note}")                                    # show it in the log
            messages.append({"role": "user", "content": note})
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