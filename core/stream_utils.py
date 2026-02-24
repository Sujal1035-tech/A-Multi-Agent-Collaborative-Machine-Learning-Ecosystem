"""
Stream Utils — Thread-safe logging via callbacks.
Uses a queue to capture logs from agent execution without hijacking sys.stdout.
"""

import json
import threading
import queue
from typing import Callable, Any, Dict, Generator


def run_handler_streaming(handler_func: Callable[[Any, Callable[[str], None]], Any], task: Any) -> Generator[str, None, None]:
    """
    Run an agent handler and yield its log output + final result as SSE events.
    
    The handler_func must accept two arguments:
    1. task: The input task object
    2. log_callback: A function that accepts a string message
    
    Usage in FastAPI:
        return StreamingResponse(run_handler_streaming(handle_analysis, task), media_type="text/event-stream")
    """
    result_holder = [None]
    error_holder = [None]
    log_queue = queue.Queue()

    def log_callback(message: str):
        """Callback to put messages into the queue."""
        log_queue.put(message)

    def worker():
        try:
            # Pass the log_callback to the handler
            result_holder[0] = handler_func(task, log_callback)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_holder[0] = str(e)
        finally:
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
