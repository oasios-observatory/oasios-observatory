#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#oasios/common/schema.py
"""Schema loading, introspection and validation for OASIOS."""

import copy
import json
import math
import threading

from pathlib import Path
from typing import Any

from jsonschema import (
    Draft7Validator,
    FormatChecker,
    SchemaError,
)

from oasios.config import settings


class SchemaConfigurationError(RuntimeError):
    """Raised when the configured JSON schema is invalid."""


class SchemaManager:
    """Thread-safe JSON Schema manager."""

    _schema: dict[str, Any] | None = None
    _schema_path: str | None = None
    _lock = threading.RLock()

    @classmethod
    def invalidate(cls) -> None:
        """Invalidate the cached schema."""

        with cls._lock:
            cls._schema = None
            cls._schema_path = None

    @classmethod
    def load(
        cls,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        """Load, validate and cache the configured schema."""

        path = Path(settings.schema_path).resolve()
        path_key = str(path)

        with cls._lock:

            if (
                not force_reload
                and cls._schema is not None
                and cls._schema_path == path_key
            ):
                return copy.deepcopy(cls._schema)

            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    schema = json.load(file)

            except (OSError, json.JSONDecodeError) as exc:
                raise SchemaConfigurationError(
                    f"Unable to load schema at {path}: {exc}"
                ) from exc

            if not isinstance(schema, dict):
                raise SchemaConfigurationError(
                    "The root JSON Schema must be an object."
                )

            try:
                Draft7Validator.check_schema(schema)

            except SchemaError as exc:
                raise SchemaConfigurationError(
                    f"Invalid JSON Schema: {exc.message}"
                ) from exc

            cls._schema = copy.deepcopy(schema)
            cls._schema_path = path_key

            return copy.deepcopy(schema)

    @classmethod
    def get_property_schema(
        cls,
        path: str,
    ) -> dict[str, Any]:
        """Retrieve a nested property definition."""

        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                "Schema property path is required."
            )

        current = cls.load()

        for part in path.split("."):

            properties = current.get("properties", {})

            if not isinstance(properties, dict):
                raise SchemaConfigurationError(
                    f"Invalid properties definition: {path}"
                )

            if part not in properties:
                raise KeyError(
                    f"Schema property not found: {path}"
                )

            current = properties[part]

            if not isinstance(current, dict):
                raise SchemaConfigurationError(
                    f"Invalid schema definition: {path}"
                )

        return copy.deepcopy(current)

    @classmethod
    def get_enum(
        cls,
        path: str,
    ) -> list[Any] | None:
        """Return permitted categorical values."""

        definition = cls.get_property_schema(path)

        values = definition.get("enum")

        if values is None:
            return None

        return copy.deepcopy(values)

    @classmethod
    def get_numeric_bounds(
        cls,
        path: str,
    ) -> tuple[float, float]:
        """Return numerical minimum and maximum."""

        definition = cls.get_property_schema(path)

        if definition.get("type") not in (
            "number",
            "integer",
        ):
            raise TypeError(
                f"{path} is not a numeric schema property."
            )

        minimum = definition.get("minimum")
        maximum = definition.get("maximum")

        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
        ):
            raise SchemaConfigurationError(
                f"Numeric property {path} requires "
                "valid minimum and maximum constraints."
            )

        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise SchemaConfigurationError(
                f"Numeric bounds for {path} must be finite."
            )

        if minimum > maximum:
            raise SchemaConfigurationError(
                f"Invalid numeric bounds for {path}."
            )

        return float(minimum), float(maximum)

    @classmethod
    def get_required_properties(
        cls,
        path: str = "",
    ) -> list[str]:
        """Return required properties of an object."""

        definition = (
            cls.get_property_schema(path)
            if path
            else cls.load()
        )

        required = definition.get("required", [])

        if not isinstance(required, list):
            raise SchemaConfigurationError(
                "Schema required property definition must be a list."
            )

        return list(required)

    @classmethod
    def validate_parameter_value(
        cls,
        path: str,
        value: Any,
    ) -> tuple[bool, list[str]]:
        """Validate one value against its schema definition."""

        definition = cls.get_property_schema(path)

        validator = Draft7Validator(
            definition,
            format_checker=FormatChecker(),
        )

        errors = sorted(
            validator.iter_errors(value),
            key=lambda error: error.message,
        )

        return (
            not errors,
            [error.message for error in errors],
        )

    @classmethod
    def validate(
        cls,
        instance: dict[str, Any],
    ) -> tuple[bool, list[str] | None]:
        """Validate a complete scenario."""

        if not isinstance(instance, dict):
            return False, [
                "Scenario must be a dictionary."
            ]

        schema = cls.load()

        validator = Draft7Validator(
            schema,
            format_checker=FormatChecker(),
        )

        errors = sorted(
            validator.iter_errors(instance),
            key=lambda error: (
                tuple(
                    str(part)
                    for part in error.absolute_path
                ),
                error.message,
            ),
        )

        if not errors:
            return True, None

        messages = []

        for error in errors:

            path = ".".join(
                str(part)
                for part in error.absolute_path
            ) or "<root>"

            messages.append(
                f"{path}: {error.message}"
            )

        return False, messages
