import requests
import json


def query_llm(
    prompt, model="gemma3:1b", temperature=0.1, base_url="http://localhost:11434"
):
    """
    Calls Ollama via web API with specified model.

    Args:
        prompt (str): The prompt to send to the model
        model (str): The model name to use (default: "gemma3:1b")
        base_url (str): The Ollama server URL (default: "http://localhost:11434")

    Returns:
        str: The model's response text

    Raises:
        requests.ConnectionError: If Ollama server is not running
        requests.RequestException: If the API request fails
        ValueError: If the response format is unexpected
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()

        if "response" not in result:
            raise ValueError(f"Unexpected response format: {result}")

        return result["response"].strip()

    except requests.ConnectionError:
        raise requests.ConnectionError(
            f"Could not connect to Ollama server at {base_url}. "
            "Please ensure Ollama is running. Start with: ollama serve"
        )
    except requests.RequestException as e:
        raise requests.RequestException(f"Ollama API request failed: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from Ollama: {e}")
