import httpx

try:
    r = httpx.get('http://localhost:5000')
    print('status', r.status_code)
    print('headers', dict(r.headers))
    print(r.text[:500])
except Exception as exc:
    print('error', exc)
