1. Who is the user, and when would they use this agent?
Software engineer who maintain the system that have to finding out the root cause of the failure, so they use this system to reduce search time

2. What is the input? What data, in what form?
Input: telemetry data (metrics, traces, logs) + query search include the time window of failure happened, and which element need to find

3. What is the output? Write the exact format: which fields, and which values are allowed.
Output: JSON include starting time, component cause the failure, and the reason of the failure
time: date and time 
component: one of frontend, checkout, payment, db, cache
reason: one of high CPU, high memory, network delay

4. What counts as correct? How do you score each part?
2 type of accuracy: 
- Strict accuracy: all of element in query search need to be correct
- Partial accuracy: at least 1 element in query search correct
- rule: time correct within 1', component and reason must exactly

5. Which mistake is the worst, and why?
Analyze wrong component 

6. Which parts don’t need an LLM and should be plain code?
- the safe code-execution tool, with time and memory limits
- checking the output format
- the step limit
- logging every step
- scoring
- generating the data

7. Which parts really need an LLM?
- Analyze query search to understand what happened and what need to do 
- Planning what to check next
- Read and analyze data
- Explain evidence

8. What’s a simple baseline to compare against? (The “dumbest” way that still produces an answer.)
- Dumb baseline: always answer the most common reason

9. Architecture
ReAct first 
