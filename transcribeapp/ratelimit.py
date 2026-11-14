from django.conf import settings

from .queue import get_redis


def client_ip(request):
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


def check_rate_limit(request, action, limit=None, window=None):
    """Increment and check a per-IP Redis counter.

    Returns ``(allowed, count)``. Raises the underlying Redis error if Redis is
    unavailable so callers can return a 503 rather than silently disabling the
    limit.
    """
    if limit is None:
        limit = settings.RATE_LIMIT_TRANSCRIPTIONS
    if window is None:
        window = settings.RATE_LIMIT_WINDOW_SECONDS

    key = f"ratelimit:{action}:{client_ip(request)}"
    r = get_redis()
    count = r.incr(key)
    if count == 1:
        r.expire(key, window)

    return count <= limit, int(count)
