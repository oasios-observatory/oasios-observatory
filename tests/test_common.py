#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for OASIOS common modules.

Run:
    pytest -v tests/test_common.py

Dependencies:
    pytest
    jsonschema
"""

import json
import sqlite3
import uuid

import pytest

from pathlib import Path
from types import SimpleNamespace

from oasios.common import consistency
from oasios.common import schema
from oasios.common import storage


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def valid_params():
    return {
        "autonomy_degree": 0.5,
        "agency_level": 0.5,
        "alignment_score": 0.5,
        "opacity": 0.5,
        "deceptiveness": 0.3,
        "initial_origin": "corporate",
        "oversight_type": "human",
        "oversight_effectiveness": "effective",
        "deployment_strategy": "public",
    }


@pytest.fixture
def valid_timeline():
    return [
        {
            "phase": "Initial Development",
            "years": "2030-2035",
        },
        {
            "phase": "Pivot",
            "years": "2040",
        },
        {
            "phase": "Mature Deployment",
            "years": "2041-2050",
        },
    ]


@pytest.fixture
def checker(valid_params, valid_timeline):
    return consistency.NarrativeChecker(
        valid_params,
        valid_timeline,
    )


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_oasios.db"

    monkeypatch.setattr(
        storage,
        "settings",
        SimpleNamespace(db_path=str(db_path)),
    )

    return db_path


@pytest.fixture
def valid_scenario():
    return {
        "id": str(uuid.uuid4()),
        "title": "Test ASI Scenario",
        "metadata": {
            "source": "generated",
            "provenance": {
                "model_used": "test-model",
            },
        },
        "origin": {},
        "architecture": {},
        "substrate": {},
        "oversight_structure": {},
        "core_capabilities": {},
        "goals_and_behavior": {},
        "impact_and_control": {},
        "scenario_content": {
            "narrative": "A test scenario.",
            "timeline": {
                "phases": [
                    {
                        "phase": "Development",
                        "years": "2030-2035",
                    }
                ]
            },
        },
        "observable_evidence": {
            "supporting_signals": [],
        },
    }


# ============================================================
# CONSISTENCY TESTS
# ============================================================

class TestNarrativeChecker:

    def test_valid_narrative(self, checker):
        valid, errors = checker.check(
            "The system developed gradually under "
            "public deployment with human oversight."
        )

        assert valid is True
        assert errors == []

    def test_empty_narrative(self, checker):
        valid, errors = checker.check("")

        assert valid is False
        assert "nonempty string" in errors[0]

    def test_non_string_narrative(self, checker):
        valid, errors = checker.check(None)

        assert valid is False
        assert errors

    def test_invalid_params_type(self):
        with pytest.raises(TypeError):
            consistency.NarrativeChecker([])

    def test_invalid_timeline_type(self, valid_params):
        with pytest.raises(TypeError):
            consistency.NarrativeChecker(
                valid_params,
                {},
            )

    def test_invalid_numeric_parameter(self, valid_params):
        valid_params["autonomy_degree"] = 1.5

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check("Test narrative.")

        assert valid is False
        assert any(
            "autonomy_degree" in error
            for error in errors
        )

    def test_boolean_rejected_as_numeric(self, valid_params):
        valid_params["autonomy_degree"] = True

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check("Test narrative.")

        assert valid is False
        assert errors

    def test_nan_rejected(self, valid_params):
        valid_params["opacity"] = float("nan")

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check("Test narrative.")

        assert valid is False
        assert errors

    def test_infinite_value_rejected(self, valid_params):
        valid_params["agency_level"] = float("inf")

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check("Test narrative.")

        assert valid is False
        assert errors

    def test_missing_required_parameter(self, valid_params):
        del valid_params["agency_level"]

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check("Test narrative.")

        assert valid is False
        assert errors

    def test_low_autonomy_conflict(self, valid_params):
        valid_params["autonomy_degree"] = 0.2

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system became fully autonomous."
        )

        assert valid is False
        assert any(
            "Low autonomy" in error
            for error in errors
        )

    def test_low_agency_conflict(self, valid_params):
        valid_params["agency_level"] = 0.1

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system controlled critical infrastructure."
        )

        assert valid is False
        assert any(
            "Low agency" in error
            for error in errors
        )

    def test_low_opacity_conflict(self, valid_params):
        valid_params["opacity"] = 0.1

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system became completely opaque."
        )

        assert valid is False
        assert errors

    def test_high_opacity_conflict(self, valid_params):
        valid_params["opacity"] = 0.9

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system became fully transparent."
        )

        assert valid is False
        assert errors

    def test_low_alignment_conflict(self, valid_params):
        valid_params["alignment_score"] = 0.1

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system remained perfectly aligned."
        )

        assert valid is False
        assert errors

    def test_high_alignment_conflict(self, valid_params):
        valid_params["alignment_score"] = 0.95

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system completely abandoned its original objectives."
        )

        assert valid is False
        assert errors

    def test_low_deceptiveness_conflict(self, valid_params):
        valid_params["deceptiveness"] = 0.1

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system habitually deceived its operators."
        )

        assert valid is False
        assert errors

    def test_high_deceptiveness_conflict(self, valid_params):
        valid_params["deceptiveness"] = 0.9

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system was completely incapable of deception."
        )

        assert valid is False
        assert errors

    def test_stealth_deployment_requirement(self, valid_params):
        valid_params["deployment_strategy"] = "stealth"

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system expanded gradually."
        )

        assert valid is False
        assert any(
            "Stealth deployment" in error
            for error in errors
        )

    def test_public_deployment_conflict(self, valid_params):
        valid_params["deployment_strategy"] = "public"

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, errors = checker.check(
            "The system was secretly deployed."
        )

        assert valid is False
        assert errors

    def test_pivot_year_conflict(
        self,
        valid_params,
        valid_timeline,
    ):
        checker = consistency.NarrativeChecker(
            valid_params,
            valid_timeline,
        )

        valid, errors = checker.check(
            "The pivot year was 2045."
        )

        assert valid is False
        assert any(
            "2040" in error
            for error in errors
        )

    def test_correct_pivot_year(
        self,
        checker,
    ):
        valid, errors = checker.check(
            "The pivot year was 2040."
        )

        assert valid is True
        assert errors == []

    def test_empty_timeline_rejected(self, valid_params):
        checker = consistency.NarrativeChecker(
            valid_params,
            [],
        )

        valid, errors = checker.check("Test narrative.")

        assert valid is False
        assert errors

    def test_malformed_timeline_rejected(self, valid_params):
        checker = consistency.NarrativeChecker(
            valid_params,
            [{"phase": "Development"}],
        )

        valid, errors = checker.check("Test narrative.")

        assert valid is False
        assert errors

    def test_checker_resets_failures(self, valid_params):
        valid_params["autonomy_degree"] = 0.1

        checker = consistency.NarrativeChecker(
            valid_params
        )

        valid, _ = checker.check(
            "The system became fully autonomous."
        )

        assert valid is False

        valid_params["autonomy_degree"] = 0.5
        checker.p["autonomy_degree"] = 0.5

        valid, errors = checker.check(
            "The system developed gradually."
        )

        assert valid is True
        assert errors == []


# ============================================================
# SCHEMA TESTS
# ============================================================

class TestSchemaManager:

    @pytest.fixture
    def schema_file(self, tmp_path, monkeypatch):

        schema_data = {
            "type": "object",
            "properties": {
                "architecture": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "transformer",
                                "hybrid",
                            ],
                        },
                        "scale": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["type"],
                }
            },
            "required": ["architecture"],
        }

        path = tmp_path / "schema.json"

        path.write_text(
            json.dumps(schema_data),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            schema,
            "settings",
            SimpleNamespace(schema_path=str(path)),
        )

        schema.SchemaManager._schema = None
        schema.SchemaManager._schema_path = None

        return path

    def test_load_schema(self, schema_file):

        result = schema.SchemaManager.load()

        assert isinstance(result, dict)
        assert result["type"] == "object"

    def test_get_property_schema(self, schema_file):

        result = schema.SchemaManager.get_property_schema(
            "architecture.type"
        )

        assert result["type"] == "string"

    def test_missing_property(self, schema_file):

        with pytest.raises(KeyError):

            schema.SchemaManager.get_property_schema(
                "architecture.missing"
            )

    def test_get_enum(self, schema_file):

        result = schema.SchemaManager.get_enum(
            "architecture.type"
        )

        assert result == [
            "transformer",
            "hybrid",
        ]

    def test_get_numeric_bounds(self, schema_file):

        result = schema.SchemaManager.get_numeric_bounds(
            "architecture.scale"
        )

        assert result == (0.0, 1.0)

    def test_invalid_numeric_property(self, schema_file):

        with pytest.raises(TypeError):

            schema.SchemaManager.get_numeric_bounds(
                "architecture.type"
            )

    def test_required_properties(self, schema_file):

        result = schema.SchemaManager.get_required_properties(
            "architecture"
        )

        assert result == ["type"]

    def test_validate_parameter_value(self, schema_file):

        valid, errors = (
            schema.SchemaManager.validate_parameter_value(
                "architecture.scale",
                0.5,
            )
        )

        assert valid is True
        assert errors == []

    def test_invalid_parameter_value(self, schema_file):

        valid, errors = (
            schema.SchemaManager.validate_parameter_value(
                "architecture.scale",
                2.0,
            )
        )

        assert valid is False
        assert errors

    def test_validate_complete_instance(self, schema_file):

        valid, errors = schema.SchemaManager.validate(
            {
                "architecture": {
                    "type": "transformer",
                    "scale": 0.5,
                }
            }
        )

        assert valid is True
        assert errors is None

    def test_invalid_complete_instance(self, schema_file):

        valid, errors = schema.SchemaManager.validate(
            {
                "architecture": {
                    "type": "invalid",
                }
            }
        )

        assert valid is False
        assert errors

    def test_missing_schema_file(
        self,
        tmp_path,
        monkeypatch,
    ):

        monkeypatch.setattr(
            schema,
            "settings",
            SimpleNamespace(
                schema_path=str(tmp_path / "missing.json")
            ),
        )

        schema.SchemaManager._schema = None
        schema.SchemaManager._schema_path = None

        with pytest.raises(
            schema.SchemaConfigurationError
        ):
            schema.SchemaManager.load()

    def test_invalid_json_schema(
        self,
        tmp_path,
        monkeypatch,
    ):

        path = tmp_path / "invalid.json"

        path.write_text(
            '{"type": "invalid-type"}',
            encoding="utf-8",
        )

        monkeypatch.setattr(
            schema,
            "settings",
            SimpleNamespace(schema_path=str(path)),
        )

        schema.SchemaManager._schema = None
        schema.SchemaManager._schema_path = None

        with pytest.raises(
            schema.SchemaConfigurationError
        ):
            schema.SchemaManager.load()


# ============================================================
# STORAGE TESTS
# ============================================================

class TestStorage:

    def test_initialize_database(self, isolated_database):

        storage.init_db()

        assert isolated_database.exists()

    def test_initialize_tables(self, isolated_database):

        storage.init_db()

        with sqlite3.connect(isolated_database) as conn:

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table'"
                )
            }

        assert "s_scenarios" in tables
        assert "ev_scenarios" in tables

    def test_invalid_table_name(self):

        with pytest.raises(ValueError):

            storage.init_table(
                "users"
            )

    def test_save_complete_scenario(
        self,
        isolated_database,
        valid_scenario,
    ):

        result = storage.save_asi_scenario(
            valid_scenario
        )

        assert result == valid_scenario["id"]

    def test_save_scenario_invalid_id(
        self,
        isolated_database,
        valid_scenario,
    ):

        valid_scenario["id"] = "invalid-id"

        with pytest.raises(ValueError):

            storage.save_asi_scenario(
                valid_scenario
            )

    def test_save_scenario_missing_title(
        self,
        valid_scenario,
    ):

        valid_scenario["title"] = ""

        with pytest.raises(ValueError):

            storage.save_asi_scenario(
                valid_scenario
            )

    def test_save_scenario_invalid_type(self):

        with pytest.raises(TypeError):

            storage.save_asi_scenario([])

    def test_save_scenario_serialization_error(
        self,
        valid_scenario,
    ):

        valid_scenario["invalid"] = object()

        with pytest.raises(ValueError):

            storage.save_asi_scenario(
                valid_scenario
            )

    def test_legacy_save(
        self,
        isolated_database,
    ):

        scenario_id = storage.save_scenario_s(
            title="Legacy scenario",
            params={"autonomy": 0.5},
            narrative="Legacy narrative.",
            timeline=[],
            model_used="test-model",
        )

        assert uuid.UUID(scenario_id)

    def test_save_evidence_scenario(
        self,
        isolated_database,
    ):

        scenario_id = storage.save_scenario_ev(
            title="Evidence scenario",
            params={},
            narrative="Evidence narrative.",
            timeline=[],
            model_used="test-model",
            signals=["signal-1"],
        )

        assert uuid.UUID(scenario_id)

    def test_get_scenarios(
        self,
        isolated_database,
    ):

        scenario_id = storage.save_scenario_s(
            title="Retrieval test",
            params={},
            narrative="Test narrative.",
            timeline=[],
            model_used="test-model",
        )

        rows = storage.get_scenarios_from_db(
            "s_scenarios"
        )

        assert len(rows) == 1
        assert rows[0]["id"] == scenario_id

    def test_delete_rows(
        self,
        isolated_database,
    ):

        scenario_id = storage.save_scenario_s(
            title="Delete test",
            params={},
            narrative="Test narrative.",
            timeline=[],
            model_used="test-model",
        )

        deleted = storage.Storage().delete_rows(
            "s_scenarios",
            "id = ?",
            (scenario_id,),
        )

        assert deleted == 1

    def test_delete_all_blocked(
        self,
        isolated_database,
    ):

        storage.save_scenario_s(
            title="Protected scenario",
            params={},
            narrative="Test narrative.",
            timeline=[],
            model_used="test-model",
        )

        deleted = storage.Storage().delete_rows(
            "s_scenarios",
            "1=1",
        )

        assert deleted == 0

    def test_invalid_column_name(
        self,
        isolated_database,
    ):

        with pytest.raises(ValueError):

            storage.Storage().get_rows_by_condition(
                "s_scenarios",
                columns=["id; DROP TABLE s_scenarios"],
            )

    def test_update_plausibility_index(
        self,
        isolated_database,
    ):

        scenario_id = storage.save_scenario_ev(
            title="Index test",
            params={},
            narrative="Test narrative.",
            timeline=[],
            model_used="test-model",
            signals=[],
        )

        storage.update_scenario_index(
            scenario_id,
            0.85,
        )

        rows = storage.get_scenarios_from_db(
            "ev_scenarios"
        )

        assert rows[0]["plausibility_index"] == 0.85

    @pytest.mark.parametrize(
        "value",
        [
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            True,
            "0.5",
        ],
    )
    def test_invalid_plausibility_index(
        self,
        value,
    ):

        with pytest.raises(ValueError):

            storage.update_scenario_index(
                str(uuid.uuid4()),
                value,
            )

    def test_next_ev_scenario_number(
        self,
        isolated_database,
    ):

        assert storage.get_next_ev_scenario_number() == 1

        storage.save_scenario_ev(
            title="EV 1",
            params={},
            narrative="Evidence narrative.",
            timeline=[],
            model_used="test-model",
            signals=[],
        )

        assert storage.get_next_ev_scenario_number() == 2

    def test_invalid_storage_table(self):

        with pytest.raises(ValueError):

            storage.get_scenarios_from_db(
                "invalid_table"
            )


# ============================================================
# REGRESSION TESTS
# ============================================================

class TestRegression:

    def test_boolean_rejected_for_storage_index(self):

        with pytest.raises(ValueError):

            storage.update_scenario_index(
                str(uuid.uuid4()),
                True,
            )

    def test_duplicate_scenario_id_rejected(
        self,
        isolated_database,
    ):

        scenario_id = str(uuid.uuid4())

        for _ in range(2):

            if _ == 0:

                storage.save_scenario_s(
                    title="Duplicate test",
                    params={},
                    narrative="Test.",
                    timeline=[],
                    model_used="test",
                    scenario_id=scenario_id,
                )

            else:

                with pytest.raises(storage.StorageError):

                    storage.save_scenario_s(
                        title="Duplicate test",
                        params={},
                        narrative="Test.",
                        timeline=[],
                        model_used="test",
                        scenario_id=scenario_id,
                    )
