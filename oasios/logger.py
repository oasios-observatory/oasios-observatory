# oasios/logger.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import structlog
import logging

structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logging.basicConfig(level="INFO")
log = structlog.get_logger()