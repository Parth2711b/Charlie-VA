import os
import asyncio
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

logger = logging.getLogger("Charlie.handler.spotify")
load_dotenv()

# We need these scopes to read the current playing track and control playback
SCOPE = "user-read-currently-playing user-read-playback-state user-modify-playback-state"

# Cache the client so we don't re-create OAuth on every single call
_spotify_client: Optional[spotipy.Spotify] = None

def get_spotify_client() -> Optional[spotipy.Spotify]:
    global _spotify_client
    if _spotify_client is not None:
        return _spotify_client
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8080/callback")
    
    if not client_id or not client_secret:
        logger.error("Spotify credentials missing in .env")
        return None
        
    try:
        # spotipy caches the token in .cache file
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SCOPE,
            open_browser=True # Opens browser for initial auth
        )
        _spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        return _spotify_client
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {e}")
        return None

async def play_song(query: str) -> str:
    """Search for a song and play it on the user's active device."""
    sp = get_spotify_client()
    if not sp:
        return "Spotify is not configured. Please check your API keys."
        
    try:
        # Search for the track
        results = sp.search(q=query, type='track', limit=1)
        
        # Fallback for strict queries if they fail (e.g. STT typos)
        if (not results or not results.get('tracks', {}).get('items', [])) and "track:" in query and "artist:" in query:
            fallback_query = query.replace("track:", "").replace("artist:", "").strip()
            logger.info("Strict search failed, falling back to: %s", fallback_query)
            results = sp.search(q=fallback_query, type='track', limit=1)
            
        if not results:
            return f"I couldn't find '{query}' on Spotify."
            
        tracks = results.get('tracks', {}).get('items', [])
        
        if not tracks:
            return f"I couldn't find '{query}' on Spotify."
            
        track = tracks[0]
        track_uri = track['uri']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        
        # Play the track
        # This will play on the currently active Spotify device
        # If no active device, it raises an exception
        sp.start_playback(uris=[track_uri])
        
        return f"Playing {track_name} by {artist_name} on Spotify."
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 404:
            return "No active Spotify device found. Please open Spotify on your phone or computer first!"
        logger.error(f"Spotify API error: {e}")
        return "I encountered an error trying to play that on Spotify."
    except Exception as e:
        logger.error(f"Spotify error: {e}")
        return "Sorry, I couldn't connect to Spotify right now."

async def pause_music() -> str:
    """Pause the current playback."""
    sp = get_spotify_client()
    if not sp:
        return "Spotify is not configured."
        
    try:
        sp.pause_playback()
        return "Paused."
    except Exception as e:
        logger.error(f"Failed to pause: {e}")
        return "I couldn't pause the music."

async def resume_music() -> str:
    """Resume playback."""
    sp = get_spotify_client()
    if not sp:
        return "Spotify is not configured."
        
    try:
        sp.start_playback()
        return "Resuming."
    except Exception as e:
        logger.error(f"Failed to resume: {e}")
        return "I couldn't resume the music."

async def skip_track() -> str:
    """Skip to next track."""
    sp = get_spotify_client()
    if not sp:
        return "Spotify is not configured."
        
    try:
        sp.next_track()
        return "Skipping to the next track."
    except Exception as e:
        logger.error(f"Failed to skip: {e}")
        return "I couldn't skip the track."

async def get_current_track() -> Optional[Dict[str, Any]]:
    """Get the currently playing track metadata for the dashboard."""
    sp = get_spotify_client()
    if not sp:
        return None
        
    try:
        current = sp.current_user_playing_track()
        if current and current.get('item'):
            item = current['item']
            return {
                "title": item['name'],
                "artist": ", ".join([a['name'] for a in item['artists']]),
                "album": item['album']['name'],
                "art_url": item['album']['images'][0]['url'] if item['album']['images'] else None,
                "is_playing": current['is_playing'],
                "progress_ms": current['progress_ms'],
                "duration_ms": item['duration_ms']
            }
        return None
    except Exception as e:
        logger.error(f"Failed to fetch current track: {e}")
        return None

# --- Background Sync Task ---
async def spotify_sync_loop():
    """Continuously poll Spotify and send metadata to the dashboard."""
    from core import websocket_bridge as ws
    
    last_track_id = None
    last_playing_state = None
    
    while True:
        try:
            # We don't want to poll too aggressively to avoid rate limits
            await asyncio.sleep(3)
            
            # Check if API keys exist before polling
            if not os.getenv("SPOTIFY_CLIENT_ID"):
                continue
                
            track_info = await get_current_track()
            if track_info:
                # Create a unique identifier for the current state
                state_id = f"{track_info['title']}_{track_info['is_playing']}"
                
                # Update dashboard if song or play state changed
                if state_id != last_track_id:
                    await ws.broadcast({
                        "type": "spotify_update",
                        "data": track_info
                    })
                    last_track_id = state_id
            else:
                # If nothing is playing but we previously showed something, clear it
                if last_track_id is not None:
                    await ws.broadcast({
                        "type": "spotify_update",
                        "data": None
                    })
                    last_track_id = None
                    
        except Exception as e:
            logger.error(f"Spotify sync loop error: {e}")
            await asyncio.sleep(5)
