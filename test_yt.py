import asyncio, httpx, re
from urllib.parse import quote_plus

async def test():
    query = 'no cap by krishna'
    url = f'https://www.youtube.com/results?search_query={quote_plus(query)}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
        r = await c.get(url, headers=headers)
        print('Status:', r.status_code)
        matches = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', r.text)
        print('Matches:', matches[:5])

asyncio.run(test())