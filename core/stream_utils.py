"""
Stream Utils — Capture print() output from agents and stream via SSE.
Runs the agent handler in a thread, captures its stdout, yields lines as SSE events.
"""

import sys
import io
import json
import threading
import queue


def run_handler_streaming(handler_func, task):
    """
    Run an agent handler and yield its print output + final result as SSE events.
    
    Usage in FastAPI:
        return StreamingResponse(run_handler_streaming(handle_analysis, task), media_type="text/event-stream")
    """
    result_holder = [None]
    error_holder = [None]
    log_queue = queue.Queue()

    def worker():
        # Redirect this thread's stdout to capture prints
        old_stdout = sys.stdout
        capture = io.StringIO()

        class TeeWriter:
            """Writes to both the original stdout AND the queue for streaming."""
            def write(self, text):
                old_stdout.write(text)  # Still print to server console
                if text.strip():  # Skip empty lines
                    log_queue.put(text.strip())
            def flush(self):
                old_stdout.flush()

        sys.stdout = TeeWriter()
        try:
            result_holder[0] = handler_func(task)
        except Exception as e:
            error_holder[0] = str(e)
        finally:
            sys.stdout = old_stdout
            log_queue.put(None)  # Signal: done

    # Start handler in background thread
    thread = threading.Thread(target=worker)
    thread.start()

    # Yield log lines as SSE events
    while True:
        try:
            msg = log_queue.get(timeout=300)  # 5 min max wait per message
            if msg is None:
                break  # Handler finished
            yield f"data: {json.dumps({'type': 'log', 'message': msg})}\n\n"
        except queue.Empty:
            break  # Timeout

    thread.join(timeout=10)

    # Yield final result or error
    if error_holder[0]:
        yield f"data: {json.dumps({'type': 'error', 'message': error_holder[0]})}\n\n"
    elif result_holder[0]:
        # Convert Pydantic model to dict
        result = result_holder[0]
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif hasattr(result, 'dict'):
            result = result.dict()
        yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
