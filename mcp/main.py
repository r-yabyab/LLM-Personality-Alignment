import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LAST_FM_API, LAST_USER
import httpx

URL = "http://ws.audioscrobbler.com/2.0/"
info = f"?method=user.getinfo&user={LAST_USER}&api_key={LAST_FM_API}&format=json"

async def main():
    async with httpx.AsyncClient() as client:
        res = await client.get(URL + info)
        # res = await client.get(URL, params={
        #     "method": "user.getinfo",
        #     "user": LAST_USER,
        #     "api_key": LAST_FM_API,
        #     "format": "json"
        # })
        data = res.json()
        print(data)

    
asyncio.run(main())