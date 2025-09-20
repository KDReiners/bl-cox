# Runbook

## Voraussetzungen
- Python 3.10
- OUTBOX_ROOT optional (Default: dynamic_system_outputs/outbox)

## Setup
Requirement already satisfied: pandas>=1.5.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 5)) (1.5.3)
Requirement already satisfied: numpy>=1.21.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 6)) (1.26.4)
Requirement already satisfied: openpyxl>=3.0.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 7)) (3.1.5)
Requirement already satisfied: scikit-learn>=1.1.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 10)) (1.4.1.post1)
Requirement already satisfied: joblib>=1.1.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 11)) (1.3.2)
Requirement already satisfied: flask>=2.0.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 14)) (2.3.3)
Requirement already satisfied: pyodbc>=4.0.34 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 17)) (4.0.39)
Requirement already satisfied: pymssql>=2.2.5 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 18)) (2.3.7)
Requirement already satisfied: sqlalchemy>=1.4.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 19)) (2.0.29)
Requirement already satisfied: tabulate>=0.9.0 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 22)) (0.9.0)
Requirement already satisfied: duckdb>=0.10.2 in /opt/homebrew/lib/python3.11/site-packages (from -r requirements.txt (line 23)) (1.4.0)
Requirement already satisfied: python-dateutil>=2.8.1 in /opt/homebrew/lib/python3.11/site-packages (from pandas>=1.5.0->-r requirements.txt (line 5)) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /opt/homebrew/lib/python3.11/site-packages (from pandas>=1.5.0->-r requirements.txt (line 5)) (2024.1)
Requirement already satisfied: et-xmlfile in /opt/homebrew/lib/python3.11/site-packages (from openpyxl>=3.0.0->-r requirements.txt (line 7)) (2.0.0)
Requirement already satisfied: scipy>=1.6.0 in /opt/homebrew/lib/python3.11/site-packages (from scikit-learn>=1.1.0->-r requirements.txt (line 10)) (1.12.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/homebrew/lib/python3.11/site-packages (from scikit-learn>=1.1.0->-r requirements.txt (line 10)) (3.4.0)
Requirement already satisfied: Werkzeug>=2.3.7 in /opt/homebrew/lib/python3.11/site-packages (from flask>=2.0.0->-r requirements.txt (line 14)) (3.0.1)
Requirement already satisfied: Jinja2>=3.1.2 in /opt/homebrew/lib/python3.11/site-packages (from flask>=2.0.0->-r requirements.txt (line 14)) (3.1.6)
Requirement already satisfied: itsdangerous>=2.1.2 in /opt/homebrew/lib/python3.11/site-packages (from flask>=2.0.0->-r requirements.txt (line 14)) (2.1.2)
Requirement already satisfied: click>=8.1.3 in /opt/homebrew/lib/python3.11/site-packages (from flask>=2.0.0->-r requirements.txt (line 14)) (8.1.7)
Requirement already satisfied: blinker>=1.6.2 in /opt/homebrew/lib/python3.11/site-packages (from flask>=2.0.0->-r requirements.txt (line 14)) (1.9.0)
Requirement already satisfied: typing-extensions>=4.6.0 in /opt/homebrew/lib/python3.11/site-packages (from sqlalchemy>=1.4.0->-r requirements.txt (line 19)) (4.14.0)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/homebrew/lib/python3.11/site-packages (from Jinja2>=3.1.2->flask>=2.0.0->-r requirements.txt (line 14)) (2.1.5)
Requirement already satisfied: six>=1.5 in /opt/homebrew/lib/python3.11/site-packages (from python-dateutil>=2.8.1->pandas>=1.5.0->-r requirements.txt (line 5)) (1.16.0)

## Konfiguration
- Gemeinsame JSONs via Submodule: 
- Wichtige Dateien:
  - data_dictionary_optimized.json
  - algorithm_config_optimized.json
  - feature_mapping.json (falls benötigt)
  - cf_cost_policy.json (Counterfactuals)

## Outbox
- Struktur:
  - outbox/churn/experiment_<id>/
  - outbox/cox/experiment_<id>/
  - outbox/counterfactuals/experiment_<id>/
- Steuerung über ENV 

## JSON-DB Import (nur json-database)

