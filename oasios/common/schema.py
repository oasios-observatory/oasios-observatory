#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/schema.py


"""Schema loading and validation for ASI scenarios."""

import json

from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, SchemaError, FormatChecker

from oasios.config import settings


class SchemaConfigurationError(RuntimeError):
    """Raised when the configured JSON schema cannot be loaded."""


class SchemaManager:
    _schema: dict[str, Any] | None = None
    _schema_path: str | None = None

    @classmethod
    def load(cls) -> dict[str, Any]:
        """Load and validate the configured JSON schema."""

        path = Path(settings.schema_path).resolve()
        path_key = str(path)

        if cls._schema is not None and cls._schema_path == path_key:
            return cls._schema

        try:
            with path.open("r", encoding="utf-8") as file:
                schema = json.load(file)

        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaConfigurationError(
                f"Unable to load JSON schema at {path}: {exc}"
            ) from exc

        if not isinstance(schema, dict):
            raise SchemaConfigurationError(
                f"JSON schema at {path} must contain an object."
            )

        try:
            Draft7Validator.check_schema(schema)

        except SchemaError as exc:
            raise SchemaConfigurationError(
                f"Invalid JSON schema at {path}: {exc.message}"
            ) from exc

        cls._schema = schema
        cls._schema_path = path_key

        return schema

    @classmethod
    def validate(
        cls,
        instance: dict[str, Any],
    ) -> tuple[bool, list[str] | None]:
        """Validate an instance and return field-specific errors."""

        schema = cls.load()

        validator = Draft7Validator(
            schema,
            format_checker=FormatChecker(),
        )

        errors = sorted(
            validator.iter_errors(instance),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )

        if not errors:
            return True, None

        messages = []

        for error in errors:
            path = ".".join(
                str(part) for part in error.absolute_path
            ) or "<root>"

            messages.append(
                f"{path}: {error.message}"
            )

        return False, messages
