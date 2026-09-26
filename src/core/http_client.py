import asyncio
import httpx

http_client: httpx.AsyncClient | None = None
send_semaphore = asyncio.Semaphore(10)


async def send_message_with_retry(
    url: str, headers: dict, json_data: dict, retries: int = 3
) -> dict | None:
    """Send a JSON payload to a remote endpoint with retry logic.

    Attempts to POST ``json_data`` to ``url`` using the configured shared
    ``http_client`` and returns the parsed JSON response body when the server
    responds with HTTP 200. If the request fails or returns a non-200 status,
    it retries up to ``retries`` times with a short delay between attempts.

    Args:
        url: The target endpoint URL.
        headers: HTTP headers to include in the request.
        json_data: JSON payload to send in the request body.
        retries: Maximum number of retry attempts after the initial request.

    Returns:
        The JSON response body as a dictionary if the request succeeds, otherwise
        ``None``.
    """
    # Fix #3: Cold Start Protection
    if http_client is None:
        print("[ERROR] http_client not initialized yet!")
        return None

    # Fix #5: Retries Logic Fix (Initial attempt + retries)
    async with send_semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await http_client.post(url, headers=headers, json=json_data)
                if response.status_code == 200:
                    # Prevent data leakage in logs
                    print(f"[REPLY SUCCESS] Status: {response.status_code}")
                    return response.json()

                print(
                    f"[REPLY ERROR] Attempt {attempt}/{retries} | Status: {response.status_code}"
                )
            except Exception as err:
                print(
                    f"[REPLY FAILED] Attempt {attempt}/{retries} | Error: {err}"
                )

            if attempt < retries:
                await asyncio.sleep(1)

    return None


# Fix #6: Missing Alias added
send_with_retry = send_message_with_retry


async def get_message_by_mid(
    url: str, headers: dict | None = None, params: dict | None = None
) -> dict | None:
    """Fetch message details by message ID (MID) with automatic URL token redaction.

    Args:
        url: Target API endpoint.
        headers: HTTP headers including authentication tokens.
        params: URL query parameters.

    Returns:
        JSON response dict if successful, otherwise empty dict or None.
    """
    # Fix #3: Cold Start Protection
    if http_client is None:
        print("[ERROR] http_client not initialized yet!")
        return None

    req_headers = headers.copy() if headers else {}
    req_params = params.copy() if params else {}

    # Fix #1: Secure Token - Extract access_token from params and move it to Authorization header
    if "access_token" in req_params:
        token = req_params.pop("access_token")
        if "Authorization" not in req_headers:
            req_headers["Authorization"] = f"Bearer {token}"

    try:
        response = await http_client.get(url, headers=req_headers, params=req_params)
        if response.status_code == 200:
            return response.json()

        print(f"[FETCH ERROR] Status: {response.status_code}")
        return {}

    except Exception as err:
        print(f"[REQUEST FAILED]: {err}")
        return {}