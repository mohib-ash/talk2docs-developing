import logging
from pathlib import Path

def setup_logging():

    # create logs folder automatically
    Path("logs").mkdir(exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(event)s | %(function)s | %(request_id)s | %(exception_type)s | %(exception)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)


    file_handler = logging.FileHandler(
        "logs/app.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("ai_saas")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()      

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False


