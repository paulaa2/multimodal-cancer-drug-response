import json
from pathlib import Path

import pandas as pd

from mcdrp.data.audit import run_audit


def test_run_audit_reports_missing_files(tmp_path: Path) -> None:
    config_path = tmp_path / "audit.json"
    config_path.write_text(
        json.dumps(
            {
                "project": "demo",
                "sources": {
                    "gdsc_response": {
                        "path": "data/raw/missing.csv",
                        "column_roles": {
                            "cell_line_name": ["CELL_LINE_NAME"],
                            "drug_name": ["DRUG_NAME"],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_audit(config_path=config_path, root=tmp_path)

    assert report["sources"]["gdsc_response"]["exists"] is False
    assert (tmp_path / "data/manifests/data_audit_report.json").exists()
    assert (tmp_path / "data/mappings/cell_line_mapping_template.csv").exists()
    assert (tmp_path / "data/mappings/drug_mapping_template.csv").exists()


def test_run_audit_detects_columns_and_writes_templates(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "CELL_LINE_NAME": ["A", "A", "B"],
            "DRUG_NAME": ["Drug 1", "Drug 2", "Drug 1"],
            "LN_IC50": [1.0, 2.0, 3.0],
        }
    ).to_csv(raw / "gdsc_response.csv", index=False)

    config_path = tmp_path / "audit.json"
    config_path.write_text(
        json.dumps(
            {
                "project": "demo",
                "sources": {
                    "gdsc_response": {
                        "path": "data/raw/gdsc_response.csv",
                        "column_roles": {
                            "cell_line_name": ["CELL_LINE_NAME"],
                            "drug_name": ["DRUG_NAME"],
                            "response": ["LN_IC50"],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_audit(config_path=config_path, root=tmp_path)

    source = report["sources"]["gdsc_response"]
    assert source["exists"] is True
    assert source["n_rows"] == 3
    assert source["column_roles"][0]["detected_column"] == "CELL_LINE_NAME"

    cell_mapping = (tmp_path / "data/mappings/cell_line_mapping_template.csv").read_text(
        encoding="utf-8"
    )
    drug_mapping = (tmp_path / "data/mappings/drug_mapping_template.csv").read_text(
        encoding="utf-8"
    )
    assert "A" in cell_mapping
    assert "Drug 1" in drug_mapping

