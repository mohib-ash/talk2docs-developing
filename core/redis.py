from fastapi import Request


async def get_redis(request: Request): 
    return request.app.state.redis

#redis for bytes!
async def get_redis_binary(request: Request):
    return request.app.state.redis_binary