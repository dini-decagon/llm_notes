# Tutorial: How `/generate_stream` Works Across Threads (asyncio for beginners)

This document explains, step by step, how a single streaming request travels through
the system, and — most importantly — **how two different threads cooperate safely** even
though `asyncio` objects are normally *not* thread-safe.

---

## 1. The 30-second mental model

When you hit `/generate_stream`, three different "execution contexts" cooperate:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CONTEXT A: the asyncio event loop  (uvicorn's main thread)               │
│  - runs your FastAPI request handler                                      │
│  - runs event_generator(), which OWNS an asyncio.Queue                    │
│  - it mostly just SLEEPS on `await queue.get()`                           │
└─────────────────────────────────────────────────────────────────────────┘
                 ▲                                    │
                 │ (tokens pushed back)               │ (request registered)
                 │                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CONTEXT B: the background worker thread  (a plain threading.Thread)      │
│  - runs requests_processing_loop() forever                                │
│  - pulls batches, asks the model for the next token                       │
│  - pushes tokens BACK into Context A's queue (the tricky part!)           │
└─────────────────────────────────────────────────────────────────────────┘
                 │                                    ▲
                 │ (batch of prompts)                 │ (one token each)
                 ▼                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│  CONTEXT C: the model worker PROCESS  (multiprocessing.Process)           │
│  - a separate OS process, talks via mp.Queue                              │
│  - actually runs the PyTorch model forward pass                           │
└─────────────────────────────────────────────────────────────────────────┘
```

The **thread question you care about is the boundary between A and B.** Context C is a
separate *process* (not a thread) and communicates with plain, thread-safe/process-safe
`multiprocessing.Queue`s, so it's much less subtle — we'll cover it briefly at the end.

---

## 2. Why is there a background thread at all?

Look at `LLMEngine.__init__` in `llm/llm.py`:

```python
# Start processing loop in a separate thread
self.thread = threading.Thread(target=self.requests_processing_loop, daemon=True)
self.thread.start()
```

The model can only generate **one token at a time per forward pass**, and it works best
when it processes a *batch* of several requests together. So the design is:

- **Web requests** arrive whenever clients connect. Each one just wants a stream of tokens.
- **The model** wants to churn continuously: grab whatever requests are active, produce one
  token for each, repeat.

These two rhythms are different. Rather than making each web request drive the model
directly, the engine runs **one long-lived background loop** (`requests_processing_loop`)
that continuously batches all active streaming requests and feeds them to the model. Web
requests just drop their prompt into a shared queue and then wait for tokens to come back.

This is a producer/consumer split:

- **Producers** = the web request handlers (they add streaming requests).
- **Consumer** = the single background thread (it processes them and streams tokens back).

`daemon=True` means: when the main program exits, this thread is killed automatically —
you don't have to join it.

---

## 3. asyncio in one paragraph (the only theory you need)

`asyncio` runs many tasks on **a single thread** using an **event loop**. A coroutine runs
until it hits `await`; at that point it *voluntarily pauses* and hands control back to the
loop, which is then free to run other tasks. When the thing it was waiting for is ready,
the loop resumes the coroutine. This is *cooperative* concurrency — nothing is truly
parallel on that thread; tasks take turns. That's why `await queue.get()` doesn't "block
the server": while one request is parked waiting for a token, the loop happily serves
other requests.

**The critical rule:** asyncio objects (like `asyncio.Queue`) are bound to *one* event
loop running on *one* thread, and they are **not thread-safe**. You must not call
`queue.put_nowait(...)` on them from a different thread. This single rule is what the whole
`/generate_stream` design has to work around.

---

## 4. Walking through one streaming request

### Step 1 — The endpoint (`main.py`)

```python
@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest, llm: LLMEngine = Depends(get_llm)):
    async def event_generator():
        loop = asyncio.get_event_loop()          # <-- grab THIS request's event loop
        async for token in llm.event_generator(loop, request.prompt):
            yield token
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

Two things to notice:

1. `asyncio.get_event_loop()` captures a reference to **the event loop the web server is
   running on**. We pass it down deliberately, because the background thread will need it
   later to "reach back" into this loop. **Hold onto this idea — it's the whole trick.**
2. `StreamingResponse` + an async generator = Server-Sent Events. Every time the generator
   `yield`s, that chunk is flushed to the HTTP client immediately.

### Step 2 — Registering the request (`llm/llm.py`, `event_generator`)

```python
async def event_generator(self, loop, prompt: str):
    asyncio.set_event_loop(loop)
    queue = asyncio.Queue()                                        # (a) client's private mailbox
    seq_id = self.workload_manager.add_streaming_request(prompt, queue, loop)  # (b)

    try:
        while True:
            data = await queue.get()                               # (c) PARK here until a token arrives
            if data is None:                                       # (d) sentinel = "stream finished"
                break
            yield f"data: {data}\n\n"                              # (e) push token to HTTP client
    finally:
        self.workload_manager.remove_finished_sequence(seq_id)     # (f) cleanup
```

- **(a)** Each request creates its **own** `asyncio.Queue`. Think of it as a private mailbox
  for this one client's tokens.
- **(b)** We hand the workload manager three things: the prompt, the queue (mailbox), **and
  the loop**. Look at `Sequence` in `workload_manager.py` — it stores `client_stream`
  (the queue) and `loop`. This is how the background thread will later find *both* the right
  mailbox and the right loop to deliver into.
- **(c)** `await queue.get()` is where this coroutine spends 99% of its life. It's parked,
  consuming no CPU, and the event loop is free to serve other requests.
- **(d)** A `None` in the mailbox is the agreed-upon "end of stream" signal (a *sentinel*).
- **(e)** Each real token is formatted as an SSE line and streamed to the client.
- **(f)** When the loop breaks (or the client disconnects), we deregister the sequence.

At this point the request is **asleep**, waiting. Nothing else happens until the background
thread produces a token. So let's switch threads.

### Step 3 — The background thread produces tokens (`requests_processing_loop`)

This runs forever on **Context B**, a completely different thread from the event loop:

```python
def requests_processing_loop(self):
    while True:
        active_sequences = self.workload_manager.get_next_batch(is_streaming=True)
        if not active_sequences:
            time.sleep(0.1)          # nothing to do; nap briefly (plain time.sleep, NOT await)
            continue

        prompts = [{'prompt': seq.prompt, 'request_id': seq.id} for seq in active_sequences]
        prompts_results = self.model_executor.execute_forward_batch(prompts)  # -> Context C

        for result in prompts_results:
            seq = self.workload_manager.get_sequence(result['request_id'])
            if result['is_finished'] or seq.token_count > self.max_tokens:
                # tell the client "we're done"
                asyncio.run_coroutine_threadsafe(seq.client_stream.put(None), seq.loop)
                seq.finished = True
                self.workload_manager.remove_finished_sequence(result['request_id'])
            else:
                # deliver one token to the client's mailbox
                asyncio.run_coroutine_threadsafe(
                    seq.client_stream.put(
                        json.dumps({"token": result['token'], "sequence_id": result['request_id']})
                    ),
                    seq.loop,
                )
                self.workload_manager.update_sequence_output(result['request_id'], result['token'])
```

Notice it uses `time.sleep(0.1)`, **not** `await asyncio.sleep(...)`. That's correct: this
is a normal thread, not a coroutine, so it uses normal blocking sleep. Using `await` here
would be a bug (there's no `async` here at all).

---

## 5. The heart of it: `run_coroutine_threadsafe`

This one line is the answer to your whole question:

```python
asyncio.run_coroutine_threadsafe(seq.client_stream.put(data), seq.loop)
```

Remember the critical rule from §3: **you cannot touch an `asyncio.Queue` from another
thread.** But the background thread (Context B) needs to put a token into a queue that
*belongs to the event loop* (Context A). If it just did `seq.client_stream.put_nowait(data)`,
it would be manipulating loop-owned state from the wrong thread → race conditions, corrupted
internal state, "got Future attached to a different loop" errors, or messages silently lost.

`asyncio.run_coroutine_threadsafe(coro, loop)` solves this. It means:

> "Hey event loop running on that *other* thread: please schedule and run this coroutine
>  (`queue.put(data)`) *on your own thread*, safely, when you get a chance."

It's the officially blessed bridge **from a normal thread → into an event loop**. The
`put` actually executes on Context A's thread, so the queue is only ever touched by its
owning loop. Thread-safety preserved. ✅

Here's the round trip, drawn out in time:

```
CONTEXT A (event loop thread)              CONTEXT B (background worker thread)
────────────────────────────────          ──────────────────────────────────────
event_generator():
  queue = asyncio.Queue()
  add_streaming_request(prompt, queue, loop)  ──(shared via WorkloadManager)──►
  await queue.get()   ← parked, idle
        .                                     get_next_batch()  → finds our sequence
        .                                     execute_forward_batch(...)  → token "cat"
        .                                     run_coroutine_threadsafe(
        .                                         queue.put("cat"), loop )
        .                                          │
   ◄────────────────────────────────────────────┘  (loop schedules the put on A)
  queue.get() returns "cat"
  yield 'data: {"token":"cat",...}'  → flushed to HTTP client
  await queue.get()   ← parked again
        .                                     ...next forward pass → token "sat"...
        .                                     run_coroutine_threadsafe(queue.put("sat"), loop)
   ◄────────────────────────────────────────────
  yield 'data: {"token":"sat",...}'
        .                                     ...model signals is_finished...
        .                                     run_coroutine_threadsafe(queue.put(None), loop)
   ◄────────────────────────────────────────────
  queue.get() returns None → break → response ends, connection closes
```

So the two threads never touch each other's data directly. They communicate through
exactly two channels:

1. **A → B:** the `WorkloadManager`'s plain `queue.Queue` (thread-safe by design) plus its
   `sequence_map`. The web handler drops a `Sequence` in; the worker picks it up.
2. **B → A:** `run_coroutine_threadsafe`, which safely injects the `queue.put(...)` back
   onto the event loop thread.

---

## 6. Why not just skip the background thread?

A fair question: why not have `event_generator` call the model itself in a loop? Because:

- The model forward pass is **CPU/GPU-heavy and synchronous**. If you ran it directly inside
  the coroutine, it would **block the event loop**, freezing *every* other request on the
  server for the duration of each token. Cooperative concurrency (§3) only works if
  coroutines don't hog the thread.
- Moving the heavy work to a separate thread (and ultimately a separate *process*) keeps the
  event loop free to do what it's good at: juggling many idle, waiting connections.

This separation — "async I/O on the event loop, heavy compute off it" — is a foundational
pattern in Python serving, and this project is a compact illustration of it.

---

## 7. The third context: the model worker process (brief)

For completeness, Context C. In `model_executor.py`:

```python
self.task_queue = mp.Queue()      # multiprocessing queue: safe ACROSS processes
self.result_queue = mp.Queue()
self.worker_process = mp.Process(target=ModelWorker.run, args=(model_name, self.task_queue, self.result_queue))
self.worker_process.start()
```

`execute_forward_batch` (called by the background thread) puts a batch on `task_queue` and
blocks on `result_queue.get()`. The child process (`ModelWorker.run` in `model_worker.py`)
loops: read a batch, run one forward pass (`generate_forward_batch`), put the tokens back.

The reason this is *easier* than the A↔B thread boundary: `multiprocessing.Queue` is
explicitly built to move data safely across the process boundary (it pickles objects and
ships them over a pipe). There's no shared-event-loop subtlety here, because the processes
share **nothing** by default. Running the model in its own process also isolates crashes:
if the model dies, the API stays up.

---

## 8. Concept cheat-sheet

| Thing in the code | What it really is | Which context |
|---|---|---|
| `event_generator` (coroutine) | async generator, parks on `await queue.get()` | A (event loop) |
| `asyncio.Queue` (`client_stream`) | per-request mailbox, loop-bound, **not** thread-safe | A (event loop) |
| `requests_processing_loop` | infinite `while True` batching loop | B (background thread) |
| `threading.Thread(daemon=True)` | the OS thread running Context B | B |
| `run_coroutine_threadsafe` | the **safe bridge B → A** | crosses B→A |
| `WorkloadManager.incoming_streaming_queue` | plain `queue.Queue`, thread-safe | shared A↔B |
| `mp.Queue` / `mp.Process` | cross-process channel + the model process | crosses B↔C |
| `None` in the mailbox | end-of-stream sentinel | A |

---

## 9. Try it yourself: watch the threads talk

The code is full of `print(...)` debug lines precisely so you can *see* this happen. Start
the server and hit the endpoint:

```bash
python main.py
# in another terminal:
curl -N -X POST http://localhost:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, I am"}'
```

Watch the server logs. You'll see the interleaving described in §5:

```
Created queue for sequence <id> in loop <L> and queue <L>
Waiting for data in queue for sequence <id>      ← Context A parks
... (background thread + model produce a token) ...
Received data in queue for sequence <id>: {"token": " a", ...}   ← delivered via bridge
Waiting for data in queue for sequence <id>      ← parks again
...
End of stream for sequence <id>                  ← the None sentinel arrived
Cleaned up sequence <id>
```

Try opening **two `curl` streams at once**. Both get parked on their own queues, the single
background thread batches them together (up to `batch_size = 4`), and each token is routed
back to the correct client via `seq.loop` + `seq.client_stream`. That "route to the right
mailbox" is exactly why each `Sequence` stores its own `loop` and `client_stream`.

---

## 10. Summary — the one thing to remember

> `/generate_stream` splits work across an **event-loop thread** (A) and a **background
> compute thread** (B). A parks on an `asyncio.Queue` doing nothing; B does the heavy model
> work and must deliver results back into A's queue. Because asyncio queues are
> **not thread-safe**, B never touches the queue directly — it uses
> **`asyncio.run_coroutine_threadsafe(queue.put(token), loop)`** to ask A's event loop to
> do the `put` on A's own thread. That single call is the safe bridge between the two
> threads, and it's the crux of the whole design.
