#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/schema.py
"""
Created on Sat Nov  8 15:35:04 2025
@author: mike
"""
import json
from jsonschema import validate, ValidationError
from oasios.config import settings

class SchemaManager:
    _schema = None

    @classmethod
    def load(cls):
        if cls._schema is None:
          with open(str(settings.schema_path)) as f:
            cls._schema = json.load(f)
        return cls._schema

    @classmethod
    def validate(cls, instance):
        try:
            validate(instance=instance, schema=cls.load())
            return True, None
        except ValidationError as e:
            return False, str(e.message)