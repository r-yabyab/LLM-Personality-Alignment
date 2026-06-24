import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from config import LAST_FM_API, LAST_FM_USER
import httpx

URL = "http://ws.audioscrobbler.com/2.0/"
get_info = f"?method=user.getinfo&user={LAST_FM_USER}&api_key={LAST_FM_API}&format=json"
get_recent_tracks_url = f"?method=user.getrecenttracks&user={LAST_FM_USER}&api_key={LAST_FM_API}&format=json"
get_top_artists_url = f"?method=user.gettopartists&user={LAST_FM_USER}&api_key={LAST_FM_API}&format=json"

mcp = FastMCP("lastfm")


@mcp.tool()
async def get_recent_tracks() -> str:
    """Get the most recently played tracks for the configured Last.fm user."""
    async with httpx.AsyncClient() as client:
        res = await client.get(URL + get_recent_tracks_url)
        data = res.json()

    if "error" in data:
        return f"Last.fm error {data['error']}: {data.get('message', '')}"

    tracks = data.get("recenttracks", {}).get("track", [])
    lines = []
    for t in tracks:
        artist = t.get("artist", {}).get("#text", "Unknown")
        name = t.get("name", "Unknown")
        lines.append(f"{artist} - {name}")
    return "\n".join(lines)


@mcp.tool()
async def get_top_artists() -> str:
    """Get the top artists for the configured Last.fm user."""
    async with httpx.AsyncClient() as client:
        res = await client.get(URL + get_top_artists_url)
        data = res.json()

    if "error" in data:
        return f"Last.fm error {data['error']}: {data.get('message', '')}"

    artists = data.get("topartists", {}).get("artist", [])
    lines = []
    for idx, artist in enumerate(artists, 1):
        name = artist.get("name", "Unknown")
        playcount = artist.get("playcount", "0")
        lines.append(f"{idx}. {name} ({playcount} plays)")
    return "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()