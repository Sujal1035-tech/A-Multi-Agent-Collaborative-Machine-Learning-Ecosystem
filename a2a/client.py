import requests
from a2a.schemas import A2ATask

import time

def send_task(url: str, task: A2ATask, timeout: int = 1200, retries: int = 3) -> dict:
    """
    Send task to agent and wait for response with auto-retry
    
    Args:
        url: Agent endpoint URL
        task: Task to send
        timeout: Request timeout in seconds (default: 1200)
        retries: Number of retries for rate limits (default: 3)
    
    Returns:
        Response dict
    """
    for attempt in range(retries + 1):
        try:
            print(f"  → Sending to {url}" + (f" (Attempt {attempt+1}/{retries+1})" if attempt > 0 else ""))
            resp = requests.post(url, json=task.dict(), timeout=timeout)
            
            # If rate limited (429), raise explicitly to catch below
            if resp.status_code == 429:
                raise requests.exceptions.HTTPError("429 Rate Limit", response=resp)
                
            resp.raise_for_status()
            print(f"  ✓ Response received")
            return resp.json()
            
        except requests.Timeout:
            print(f"  ✗ Timeout after {timeout}s")
            raise Exception(f"Request to {url} timed out after {timeout}s")
            
        except requests.exceptions.HTTPError as e:
            # Handle Rate Limit (429) specifically
            if e.response.status_code == 429:
                wait_time = 60 * (attempt + 1)  # Progressive wait: 60s, 120s, 180s...
                print(f"  ⚠️ Rate Limit Hit! Pausing for {wait_time}s to cooldown...")
                time.sleep(wait_time)
                continue  # Retry
            
            # Handle Server Error (500) which might be temporary
            if e.response.status_code >= 500 and attempt < retries:
                wait_time = 5
                print(f"  ⚠️ Server Error {e.response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            print(f"  ✗ Request failed: {e}")
            raise
            
        except requests.RequestException as e:
            print(f"  ✗ Request failed: {e}")
            raise
