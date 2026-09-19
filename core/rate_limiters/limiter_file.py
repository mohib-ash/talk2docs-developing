from slowapi import Limiter
from slowapi.util import get_remote_address


limiter = Limiter(
    key_func=get_remote_address,
    application_limits=[],
    headers_enabled=True,
    strategy=None, 
    storage_uri="redis://localhost:6379",
    storage_options={},
    auto_check=True,
    swallow_errors=False,
    in_memory_fallback=["1/minute"],
    in_memory_fallback_enabled=True,
    retry_after="delta",
)