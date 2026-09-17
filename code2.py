import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI, Response, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Scope, Receive, Send
from redis.asyncio import Redis

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from core.rate_limiters.limiter_file import limiter

from core.Exceptions.exception_handlers import global_exception_handler, unexpected_exception_handler
from core.Exceptions.exceptions import AppException

from routers.Ai import take_doc_route
from routers.auth import auth_route
from routers.users import users_routes
from routers.chat import question_route 

from utils.logging.config import setup_logging
from utils.config import settings

setup_logging() 

os.environ["LANGCHAIN_TRACING_V2"] = "false"


#Pure ASGI Middleware for Process Time & Request Logging 
class ProcessTimeMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        path = scope.get("path", "")
        method = scope.get("method", "")
        
        client = scope.get("client")
        client_host = client[0] if client else None

        print(f"[REQUEST START] {method} {path} client={client_host}")

        status_code = 500

        async def wrapped_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                process_time_ms = (time.perf_counter() - start_time) * 1000
                
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time-ms", f"{process_time_ms:.2f} ms".encode()))
                message["headers"] = headers
                
                print(f"[REQUEST END] {method} {path} status={status_code} time={process_time_ms:.2f}ms")
            
            await send(message)

        await self.app(scope, receive, wrapped_send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application")
    
    
    app.state.redis = Redis(
        host="localhost",
        port=6379,
        decode_responses=True  # keeps str str!
    )
    
    
    app.state.redis_binary = Redis(
        host="localhost",
        port=6379,
        decode_responses=False 
    )
    
    yield
    
    print("Closing application")
    await app.state.redis.close()
    await app.state.redis_binary.close()  

app = FastAPI(
    title="Social Network Aggregator API",
    lifespan=lifespan
)


app.add_middleware(ProcessTimeMiddleware)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://10.0.2.2:8000",
    "https://www.yourdomain.com",
    "https://yourdomain.com",
    "https://staging.yourdomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    print(f"--> Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"<-- Outgoing response: {response.status_code}")
    return response


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(AppException, global_exception_handler)
app.add_exception_handler(Exception, unexpected_exception_handler)

# Routers
app.include_router(router=users_routes.router)
app.include_router(router=auth_route.router)
app.include_router(router=take_doc_route.router)
app.include_router(router=question_route.router)


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {"message": "Hello World"}