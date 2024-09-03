import logging as lg
import os
import sys
from datetime import datetime, timezone

IS_DEBUG = os.getenv("DEBUG", "false").lower() in ("yes", "true", "t", "1")
LOG_PRESIS = os.getenv("LOG_PRESIS", "false").lower() in ("yes", "true", "t", "1")
logging = lg.getLogger(__name__)
logging.setLevel(lg.DEBUG if IS_DEBUG else lg.INFO)
formatter = lg.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handlers = []
if LOG_PRESIS:
    handler = lg.FileHandler(
        datetime.now(timezone.utc).strftime("%Y-%m-%d.log"), encoding="utf-8"
    )
    handler.setLevel(lg.DEBUG if IS_DEBUG else lg.INFO)
    handler.setFormatter(formatter)
    handlers.append(handler)

    handler = lg.StreamHandler(sys.stdout)
    handler.setLevel(lg.INFO)
    handler.setFormatter(formatter)
    handlers.append(handler)
else:
    handler = lg.StreamHandler(sys.stdout)
    handler.setLevel(lg.DEBUG if IS_DEBUG else lg.INFO)
    handler.setFormatter(formatter)
    handlers.append(handler)
for handler in handlers:
    logging.addHandler(handler)
