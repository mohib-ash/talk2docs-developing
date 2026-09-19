class RateLimits:

    class Auth:
        LOGIN = "5/minute"
        REGISTER = "3/minute"
        LOGOUT = "10/minute"
        LOGOUT_ALL = "3/minute"
        CHANGE_PASSWORD = "3/minute"

    class User:
        READ = "60/minute"
        WRITE = "20/minute"

    class Posts:
        READ = "60/minute"
        WRITE = "20/minute"
        UPDATE = "20/minute"
        DELETE = "10/minute"

    class Comments:
        READ = "120/minute"
        WRITE = "30/minute"
        DELETE = "20/minute"

    class Likes:
        TOGGLE = "30/minute"
        READ = "60/minute"

    class AI:
        DEFAULT = "30/minute"
        FILE_UPLOAD = "30/hour"
        ASK_QUESTION = "40/minute"

    class Admin:
        READ = "10/minute"
        WRITE = "5/minute"
    
    class Session:
        COUNT_ACTIVE = "30/minute"
        REVOKE_CURRENT = "10/minute"
        