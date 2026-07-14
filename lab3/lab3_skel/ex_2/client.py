import time
import asyncio

import requests
import httpx


URL = "http://localhost:8000"
TIMEOUT_SECONDS = 10.0


def fetch_sync():
    request = requests.get(URL, timeout=TIMEOUT_SECONDS)
    return request


async def fetch_async():
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.get(URL)
        return response


def run_sync():
    start = time.perf_counter()

    # 3 apeluri
    requests = [fetch_sync() for _ in range(3)]

    end = time.perf_counter()

    print(f"Execution time: {end - start:.2f} seconds")
    print(f"Responses: {[r.json() for r in requests]}")


async def run_async():
    start = time.perf_counter()

    # asyncio.gather(...)
    #responses = await asyncio.gather(*[fetch_async() for _ in range(3)])
    responses = await asyncio.gather(
        fetch_async(), 
        fetch_async(),
        fetch_async()
    )

    end = time.perf_counter()

    print(f"Execution time: {end - start:.2f} seconds")
    print(f"Responses: {[r.json() for r in responses]}")


if __name__ == "__main__":
    run_sync()

    asyncio.run(
        run_async()
    )