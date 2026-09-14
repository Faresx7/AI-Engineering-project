import asyncio
import httpx

http_client: httpx.AsyncClient | None = None
send_semaphore = asyncio.Semaphore(10)


async def send_with_retry(
    url: str, headers: dict, json_data: dict, retries: int = 3
                        ) -> dict | None:
    
    if http_client is None:
        print("[ERROR] http_client not initialized yet!")
        return None

    async with send_semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await http_client.post(url, headers=headers, json=json_data)
                if response.status_code == 200:
                    print(f"[REPLY SUCCESS] {response.json()}")
                    return response.json()
                print(f"[REPLY ERROR] Attempt {attempt}/{retries} | {response.status_code} | {response.json()}")
            except Exception as err:
                print(f"[REPLY FAILED] Attempt {attempt}/{retries} | {err}")

            if attempt < retries:
                await asyncio.sleep(1)
    return None


async def get_message_by_mid(url: str, params: dict):
    if http_client is None:
            print("[ERROR] http_client not initialized yet!")
            return None
    
    try:
        response = await http_client.get(url, params=params)
        if response.status_code == 200:
            return response.json()

        print(f"[FETCH ERROR] Status: {response.status_code} | Body: {response.json()}")
        return {}
    
    except Exception as err:
        print(f"[REQUEST FAILED]: {err}")
        return {}