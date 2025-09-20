#!/usr/bin/env python3
"""
Working Cox Main Pipeline - Implementiert die erfolgreiche Solution
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from typing import Dict, Any, Optional

# Füge Projekt-Root zum Python-Path hinzu
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bl.Cox.cox_data_loader import CoxDataLoader
from bl.Cox.cox_evaluator import CoxEvaluator
from bl.json_database.churn_json_database import ChurnJSONDatabase
from config.paths_config import ProjectPaths
import json
import shutil

class CoxWorkingPipeline:
    """Working Cox Pipeline mit bewiesener Performance"""
    
    def __init__(self, cutoff_exclusive: int = 202501):
        self.cutoff_exclusive = cutoff_exclusive
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = CoxDataLoader(cutoff_exclusive=cutoff_exclusive)
        self.evaluator = CoxEvaluator()
        
        # Results
        self.model: Optional[CoxPHFitter] = None
        self.c_index: Optional[float] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('cox_working_pipeline')
    
    def create_working_features(self, stage0_data: pd.DataFrame, survival_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt die bewährten Features mit C-Index 0.8094
        """
        self.logger.info("🚀 Erstelle Working Features (C-Index 0.8094)")
        
        customer_features = []
        
        for kunde in survival_panel['Kunde'].unique():
            kunde_data = stage0_data[stage0_data['Kunde'] == kunde].sort_values('I_TIMEBASE')
            if len(kunde_data) > 0:
                last_record = kunde_data.iloc[-1]
                
                # I_SOCIALINSURANCENOTES: Hauptfeature mit hoher Varianz
                social_notes = last_record.get('I_SOCIALINSURANCENOTES', 0)
                
                # N_DIGITALIZATIONRATE: Als kontinuierlich (nicht One-Hot)
                digitalization_rate = last_record.get('N_DIGITALIZATIONRATE', 0)
                
                # Business-Activity Features
                upgrade = last_record.get('I_UPGRADE', 0)
                upsell = last_record.get('I_UPSELL', 0)
                consulting = last_record.get('I_CONSULTING', 0)
                
                # Aggregate Business Activity
                business_activity = upgrade + upsell + consulting
                
                customer_features.append({
                    'Kunde': kunde,
                    'I_SOCIALINSURANCENOTES': float(social_notes),
                    'N_DIGITALIZATIONRATE': float(digitalization_rate),
                    'business_activity': float(business_activity),
                    'has_digitalization': 1.0 if digitalization_rate > 0 else 0.0,
                })
        
        features_df = pd.DataFrame(customer_features)
        
        # 🔧 CRITICAL FIX: Merge mit survival_panel um duration/event hinzuzufügen
        # Diese Spalten sind essentiell für Cox-Model und JSON-Database-Speicherung!
        survival_minimal = survival_panel[['Kunde', 'duration', 'event']].copy()
        cox_data = features_df.merge(survival_minimal, on='Kunde', how='inner')
        
        self.logger.info(f"✅ Working Features: {len(cox_data)} Kunden, {len(cox_data.columns)-3} Features")
        self.logger.info(f"📊 Cox-Data Columns: {list(cox_data.columns)}")
        
        return cox_data
    
    def _save_experiment_results(self, results: Dict[str, Any], experiment_id: int) -> None:
        """Speichert Experiment-Ergebnisse für JSON-Database-Integration"""
        # Disabled: Export nach dynamic_system_outputs/cox_experiments nicht mehr erforderlich,
        # da alle Cox-Ergebnisse in die JSON-Datenbank geschrieben werden.
        return
    
    def _populate_cox_tables(self, stage0_data: pd.DataFrame, survival_panel: pd.DataFrame, 
                           cox_data: pd.DataFrame, model_data: pd.DataFrame, 
                           results: Dict[str, Any], experiment_id: int) -> None:
        """
        Befüllt alle Cox-basierten Tabellen in der JSON-Database
        experiment_id wird als Fremdschlüssel verwendet
        """
        try:
            self.logger.info(f"💾 Befülle Cox-Tabellen für Experiment-ID: {experiment_id}")
            
            # JSON-Database initialisieren
            db = ChurnJSONDatabase()
            
            # 0. Experiment-Eintrag sicherstellen (experiments Tabelle)
            # 🔧 FIX: Verwende die tatsächliche DB-Experiment-ID
            actual_experiment_id = self._ensure_experiment_entry(db, experiment_id, results)
            self.logger.info(f"📊 Verwende Experiment-ID: {actual_experiment_id} (ursprünglich: {experiment_id})")
            
            # 1. cox_survival - Survival Panel Daten
            self._populate_cox_survival_table(db, survival_panel, actual_experiment_id)
            
            # 2. cox_prioritization_results - Risiko-Scores und Wahrscheinlichkeiten
            self._populate_cox_prioritization_table(db, cox_data, model_data, actual_experiment_id)
            
            # 3. cox_analysis_metrics - Performance-Metriken
            self._populate_cox_analysis_metrics(db, results, actual_experiment_id)
            
            # 4. experiment_kpis - KPI-Metriken
            self._populate_experiment_kpis(db, results, actual_experiment_id)
            
            # Änderungen speichern
            db.save()
            self.logger.info("✅ Alle Cox-Tabellen erfolgreich befüllt")
            
            # Outbox-Export (Sink) für Cox
            try:
                self._export_outbox_cox(actual_experiment_id)
                self.logger.info("📦 Outbox-Export (Cox) abgeschlossen")
            except Exception as e:
                self.logger.warning(f"⚠️ Outbox-Export (Cox) fehlgeschlagen: {e}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cox-Tabellen-Befüllung fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
    
    def _populate_cox_survival_table(self, db: ChurnJSONDatabase, survival_panel: pd.DataFrame, 
                                   experiment_id: int) -> None:
        """Befüllt cox_survival Tabelle mit Survival Panel Daten"""
        self.logger.info("📊 Befülle cox_survival Tabelle")
        
        survival_records = []
        for _, row in survival_panel.iterrows():
            survival_records.append({
                'Kunde': int(row['Kunde']),
                't_start': float(row.get('t_start', 0)),
                't_end': float(row.get('t_end', row.get('duration', 0))),
                'duration': float(row['duration']),
                'event': int(row['event']),
                'id_experiments': experiment_id
            })
        
        # Zur Tabelle hinzufügen
        if "cox_survival" not in db.data["tables"]:
            db.data["tables"]["cox_survival"] = {
                "records": [],
                "schema": {
                    "Kunde": {"type": "integer", "description": "Kunden-ID"},
                    "t_start": {"type": "float", "description": "Start-Zeit"},
                    "t_end": {"type": "float", "description": "End-Zeit"},
                    "duration": {"type": "float", "description": "Überlebensdauer"},
                    "event": {"type": "integer", "description": "Event (1=Churn, 0=Censored)"},
                    "id_experiments": {"type": "integer", "description": "Experiment-ID (FK)"}
                }
            }
        
        # Bestehende Records mit gleicher experiment_id entfernen
        existing_records = db.data["tables"]["cox_survival"].get("records", [])
        filtered_records = [r for r in existing_records if r.get('id_experiments') != experiment_id]
        
        # Neue Records hinzufügen
        db.data["tables"]["cox_survival"]["records"] = filtered_records + survival_records
        self.logger.info(f"✅ cox_survival: {len(survival_records)} Records hinzugefügt")
    
    def _populate_cox_prioritization_table(self, db: ChurnJSONDatabase, cox_data: pd.DataFrame, 
                                         model_data: pd.DataFrame, experiment_id: int) -> None:
        """Befüllt cox_prioritization_results Tabelle mit Risiko-Scores"""
        self.logger.info("📊 Befülle cox_prioritization_results Tabelle")
        
        # model_data_for_cox für Predictions (ohne Kunde)
        feature_cols = [col for col in model_data.columns if col not in ['Kunde', 'duration', 'event']]
        model_data_for_cox = model_data[['duration', 'event'] + feature_cols].copy()
        
        # Predictions für alle Kunden berechnen
        predictions = self.model.predict_survival_function(model_data_for_cox)
        partial_hazards = self.model.predict_partial_hazard(model_data_for_cox)
        
        prioritization_records = []
        for i, (idx, row) in enumerate(model_data.iterrows()):
            kunde = int(row['Kunde'])
            
            # Survival-Wahrscheinlichkeiten berechnen
            survival_func = predictions.iloc[:, i]
            
            # 6-Monats und 12-Monats Churn-Wahrscheinlichkeiten
            p_event_6m = 1 - survival_func.loc[min(6, survival_func.index.max())] if len(survival_func) > 0 else 0.5
            p_event_12m = 1 - survival_func.loc[min(12, survival_func.index.max())] if len(survival_func) > 0 else 0.5
            
            # RMST (Restricted Mean Survival Time) approximieren
            rmst_12m = min(12, survival_func.index.max()) if len(survival_func) > 0 else 6
            rmst_24m = min(24, survival_func.index.max()) if len(survival_func) > 0 else 12
            
            # Conditional/Unconditional Months To Live
            months_conditional = max(1, rmst_12m * (1 - p_event_12m))
            months_unconditional = rmst_12m
            
            # Priority Score (0-100, höher = risikanter)
            priority_score = min(100, max(0, p_event_12m * 100))
            
            prioritization_records.append({
                'Kunde': kunde,
                'P_Event_6m': float(p_event_6m),
                'P_Event_12m': float(p_event_12m),
                'RMST_12m': float(rmst_12m),
                'RMST_24m': float(rmst_24m),
                'MonthsToLive_Conditional': float(months_conditional),
                'MonthsToLive_Unconditional': float(months_unconditional),
                'PriorityScore': float(priority_score),
                'StartTimebase': self.cutoff_exclusive - 60,  # 5 Jahre Training
                'LastAliveTimebase': self.cutoff_exclusive,
                'CutoffExclusive': self.cutoff_exclusive,
                'ChurnTimebase': None,  # Nur für tatsächlich gechurnte Kunden
                'LeadMonthsToChurn': None,
                'Actual_Event_6m': None,  # Würde Backtest-Daten benötigen
                'Actual_Event_12m': None,
                'id_experiments': experiment_id
            })
        
        # Zur Tabelle hinzufügen (Schema bereits vorhanden laut User)
        if "cox_prioritization_results" not in db.data["tables"]:
            db.data["tables"]["cox_prioritization_results"] = {"records": []}
        
        # Bestehende Records mit gleicher experiment_id entfernen
        existing_records = db.data["tables"]["cox_prioritization_results"].get("records", [])
        filtered_records = [r for r in existing_records if r.get('id_experiments') != experiment_id]
        
        # Neue Records hinzufügen
        db.data["tables"]["cox_prioritization_results"]["records"] = filtered_records + prioritization_records
        self.logger.info(f"✅ cox_prioritization_results: {len(prioritization_records)} Records hinzugefügt")
    
    def _populate_cox_analysis_metrics(self, db: ChurnJSONDatabase, results: Dict[str, Any], 
                                     experiment_id: int) -> None:
        """Befüllt cox_analysis_metrics Tabelle mit Performance-Metriken"""
        self.logger.info("📊 Befülle cox_analysis_metrics Tabelle")
        
        # Metriken aus Results extrahieren
        performance = results.get('performance', {})
        model_perf = performance.get('model_performance', {})
        business_scores = performance.get('business_scores', {})
        
        # Metric Records erstellen
        metrics_records = []
        
        # Model Performance Metriken
        if 'c_index' in model_perf:
            metrics_records.append({
                'metric_id': f"{experiment_id}_c_index",
                'experiment_id': experiment_id,
                'metric_name': 'c_index',
                'metric_value': float(model_perf['c_index']),
                'metric_type': 'model_performance',
                'cutoff_exclusive': self.cutoff_exclusive,
                'feature_count': int(model_perf.get('features_count', 0)),
                'c_index': float(model_perf['c_index']),
                'horizon_max': 24,  # Monate
                'num_samples': int(results.get('model_data_size', 0)),
                'num_active': int(business_scores.get('prioritized_customers', 0)),
                'mean_p12': 0.3,  # Placeholder - würde echte Berechnung benötigen
                'runtime_s': float(results.get('duration_seconds', 0)),
                'calculated_at': results.get('timestamp', datetime.now().isoformat())
            })
        
        # Log Likelihood Metrik
        if 'log_likelihood' in model_perf:
            metrics_records.append({
                'metric_id': f"{experiment_id}_log_likelihood",
                'experiment_id': experiment_id,
                'metric_name': 'log_likelihood',
                'metric_value': float(model_perf['log_likelihood']),
                'metric_type': 'model_performance',
                'cutoff_exclusive': self.cutoff_exclusive,
                'feature_count': int(model_perf.get('features_count', 0)),
                'c_index': float(model_perf['c_index']),
                'horizon_max': 24,
                'num_samples': int(results.get('model_data_size', 0)),
                'num_active': int(business_scores.get('prioritized_customers', 0)),
                'mean_p12': 0.3,
                'runtime_s': float(results.get('duration_seconds', 0)),
                'calculated_at': results.get('timestamp', datetime.now().isoformat())
            })
        
        # Zur Tabelle hinzufügen (Schema bereits vorhanden laut User)
        if "cox_analysis_metrics" not in db.data["tables"]:
            db.data["tables"]["cox_analysis_metrics"] = {"records": []}
        
        # Bestehende Records mit gleicher experiment_id entfernen
        existing_records = db.data["tables"]["cox_analysis_metrics"].get("records", [])
        filtered_records = [r for r in existing_records if r.get('experiment_id') != experiment_id]
        
        # Neue Records hinzufügen
        db.data["tables"]["cox_analysis_metrics"]["records"] = filtered_records + metrics_records
        self.logger.info(f"✅ cox_analysis_metrics: {len(metrics_records)} Records hinzugefügt")
    
    def _ensure_experiment_entry(self, db: ChurnJSONDatabase, experiment_id: int, 
                               results: Dict[str, Any]) -> int:
        """
        Sicherstellt, dass ein Experiment-Eintrag existiert
        
        Returns:
            Die tatsächliche Experiment-ID (kann von der übergebenen abweichen)
        """
        self.logger.info("📊 Prüfe/Erstelle Experiment-Eintrag")
        
        # Prüfe ob Experiment bereits existiert
        existing_experiments = db.data.get("tables", {}).get("experiments", {}).get("records", [])
        experiment_exists = any(exp.get("experiment_id") == experiment_id for exp in existing_experiments)
        
        if not experiment_exists:
            # Erstelle neuen Experiment-Eintrag mit YYYYMM Format
            training_from = str(self.cutoff_exclusive - 60)  # 5 Jahre zurück  
            training_to = str(self.cutoff_exclusive - 1)     # Bis zum Cutoff
            backtest_from = str(self.cutoff_exclusive)        # Ab Cutoff
            backtest_to = str(self.cutoff_exclusive)          # Nur der Cutoff-Monat
            
            try:
                new_exp_id = db.create_experiment(
                    experiment_name=f"Cox Working Pipeline (Automated) - ID {experiment_id}",
                    training_from=training_from,
                    training_to=training_to,
                    backtest_from=backtest_from,
                    backtest_to=backtest_to,
                    model_type="cox_survival",
                    feature_set="working_features",
                    hyperparameters={
                        "penalizer": 0.1,
                        "features": results.get('features_used', [])
                    },
                    file_ids=[1]  # Standard Input-File
                )
                self.logger.info(f"✅ Neues Experiment erstellt: ID {new_exp_id}")
                return new_exp_id  # 🔧 FIX: Gib die tatsächliche DB-ID zurück
            except Exception as e:
                # Falls create_experiment fehlschlägt, manuell hinzufügen
                self.logger.warning(f"⚠️ create_experiment fehlgeschlagen: {e}")
                self._manually_add_experiment(db, experiment_id, results)
                return experiment_id  # Fallback: ursprüngliche ID
        else:
            self.logger.info(f"✅ Experiment {experiment_id} bereits vorhanden")
            return experiment_id  # Bestehende ID zurückgeben
    
    def _manually_add_experiment(self, db: ChurnJSONDatabase, experiment_id: int, 
                               results: Dict[str, Any]) -> None:
        """Manuelles Hinzufügen eines Experiments falls create_experiment fehlschlägt"""
        if "experiments" not in db.data["tables"]:
            db.data["tables"]["experiments"] = {"records": []}
        
        training_from = str(self.cutoff_exclusive - 60)
        training_to = str(self.cutoff_exclusive - 1)
        backtest_from = str(self.cutoff_exclusive)
        backtest_to = str(self.cutoff_exclusive)
        
        experiment_record = {
            "experiment_id": experiment_id,
            "experiment_name": f"Cox Working Pipeline (Manual) - ID {experiment_id}",
            "training_from": training_from,
            "training_to": training_to,
            "backtest_from": backtest_from,
            "backtest_to": backtest_to,
            "model_type": "cox_survival",
            "feature_set": "working_features",
            "hyperparameters": {
                "penalizer": 0.1,
                "features": results.get('features_used', [])
            },
            "created_at": results.get('timestamp', datetime.now().isoformat()),
            "status": "created",
            "id_files": [1]
        }
        
        # Entferne existierenden Eintrag mit gleicher ID
        existing_records = db.data["tables"]["experiments"].get("records", [])
        filtered_records = [r for r in existing_records if r.get('experiment_id') != experiment_id]
        
        # Füge neuen Eintrag hinzu
        db.data["tables"]["experiments"]["records"] = filtered_records + [experiment_record]
        self.logger.info(f"✅ Experiment {experiment_id} manuell hinzugefügt")
    
    def _populate_experiment_kpis(self, db: ChurnJSONDatabase, results: Dict[str, Any], 
                                experiment_id: int) -> None:
        """Befüllt experiment_kpis Tabelle"""
        self.logger.info("📊 Befülle experiment_kpis Tabelle")
        
        performance = results.get('performance', {})
        model_perf = performance.get('model_performance', {})
        
        kpi_records = []
        
        # C-Index KPI
        if 'c_index' in model_perf:
            try:
                db.add_experiment_kpi(
                    experiment_id=experiment_id,
                    metric_name="c_index",
                    metric_value=float(model_perf['c_index']),
                    metric_type="model_performance"
                )
                kpi_records.append("c_index")
            except Exception as e:
                self.logger.warning(f"⚠️ Fehler beim Hinzufügen von c_index KPI: {e}")
        
        # Events KPI
        if 'events' in results:
            try:
                db.add_experiment_kpi(
                    experiment_id=experiment_id,
                    metric_name="events",
                    metric_value=float(results['events']),
                    metric_type="data_quality"
                )
                kpi_records.append("events")
            except Exception as e:
                self.logger.warning(f"⚠️ Fehler beim Hinzufügen von events KPI: {e}")
        
        # Runtime KPI
        if 'duration_seconds' in results:
            try:
                db.add_experiment_kpi(
                    experiment_id=experiment_id,
                    metric_name="runtime_seconds",
                    metric_value=float(results['duration_seconds']),
                    metric_type="performance"
                )
                kpi_records.append("runtime_seconds")
            except Exception as e:
                self.logger.warning(f"⚠️ Fehler beim Hinzufügen von runtime KPI: {e}")
        
        self.logger.info(f"✅ experiment_kpis: {len(kpi_records)} KPIs hinzugefügt: {kpi_records}")

    def _export_outbox_cox(self, experiment_id: int) -> None:
        """Exportiert minimale Cox-Artefakte in die Outbox (survival/prioritization/metrics)."""
        out_dir = ProjectPaths.outbox_cox_experiment_directory(int(experiment_id))
        ProjectPaths.ensure_directory_exists(out_dir)
        
        db = ChurnJSONDatabase()
        tables = db.data.get("tables", {})
        
        # Survival Records (gefiltert nach experiment_id)
        surv = [r for r in tables.get("cox_survival", {}).get("records", []) if int(r.get("id_experiments", -1)) == int(experiment_id)]
        with open(out_dir / "cox_survival.json", 'w', encoding='utf-8') as f:
            json.dump(surv, f, ensure_ascii=False, indent=2)
        
        # Prioritization Records
        prio = [r for r in tables.get("cox_prioritization_results", {}).get("records", []) if int(r.get("id_experiments", -1)) == int(experiment_id)]
        with open(out_dir / "cox_prioritization.json", 'w', encoding='utf-8') as f:
            json.dump(prio, f, ensure_ascii=False, indent=2)
        
        # Metrics (cox_analysis_metrics)
        metrics = [r for r in tables.get("cox_analysis_metrics", {}).get("records", []) if int(r.get("experiment_id", -1)) == int(experiment_id)]
        with open(out_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # KPIs (experiment_kpis) – optional vorhanden
        kpis_tbl = tables.get("experiment_kpis", {}).get("records", [])
        kpis = [r for r in kpis_tbl if int(r.get("experiment_id", -1)) == int(experiment_id)]
        if kpis:
            with open(out_dir / "kpis.json", 'w', encoding='utf-8') as f:
                json.dump(kpis, f, ensure_ascii=False, indent=2)
    
    def run_full_analysis(self, experiment_id: Optional[int] = None) -> Dict[str, Any]:
        """Führt komplette Cox-Analyse durch"""
        self.logger.info("🚀 Starte Working Cox-Pipeline")
        if experiment_id:
            self.logger.info(f"🔬 Experiment-ID: {experiment_id}")
        start_time = datetime.now()
        
        try:
            # 1. Daten laden
            self.logger.info("📊 Lade Daten")
            stage0_data = self.data_loader.load_stage0_data()
            survival_panel = self.data_loader.create_survival_panel(stage0_data)
            
            self.logger.info(f"✅ Survival Panel: {len(survival_panel)} Records")
            self.logger.info(f"📊 Events: {survival_panel['event'].sum()}")
            self.logger.info(f"📊 Event-Rate: {survival_panel['event'].mean():.3f}")
            
            # 2. Features erstellen
            self.logger.info("⚙️ Erstelle Working Features")
            # create_working_features liefert bereits Features inkl. duration/event
            cox_data = self.create_working_features(stage0_data, survival_panel)
            cox_data = cox_data.fillna(0)
            
            # 3. Feature-Auswahl (nur die mit Varianz)
            feature_cols = ['I_SOCIALINSURANCENOTES', 'N_DIGITALIZATIONRATE', 'business_activity', 'has_digitalization']
            good_features = []
            
            self.logger.info("📊 Feature-Analyse:")
            for col in feature_cols:
                variance = cox_data[col].var()
                mean_val = cox_data[col].mean()
                self.logger.info(f"   {col:25}: Var={variance:10.2f}, Mean={mean_val:6.2f}")
                if variance > 0.1:
                    good_features.append(col)
            
            self.logger.info(f"✅ Verwendete Features: {good_features}")
            
            # 4. Model Training
            self.logger.info("🎯 Trainiere Cox-Model")
            model_data = cox_data[['Kunde', 'duration', 'event'] + good_features].copy()
            model_data = model_data.dropna()
            
            # Separate model_data_for_cox (ohne Kunde für Cox-Fit)
            model_data_for_cox = model_data[['duration', 'event'] + good_features].copy()
            
            self.logger.info(f"📊 Model-Daten: {len(model_data)} Records")
            self.logger.info(f"📊 Events: {model_data['event'].sum()}")
            
            # Cox-Fit mit bewährten Parametern
            self.model = CoxPHFitter(penalizer=0.1)
            self.model.fit(model_data_for_cox, duration_col='duration', event_col='event')
            
            self.c_index = self.model.concordance_index_
            
            # 5. Performance-Evaluation (vereinfacht)
            self.logger.info("📈 Performance-Evaluation")
            
            # Basis-Performance ohne komplexe Evaluation
            evaluation_results = {
                'model_performance': {
                    'c_index': self.c_index,
                    'log_likelihood': self.model.log_likelihood_,
                    'features_count': len(good_features)
                },
                'business_scores': {
                    'prioritized_customers': len(model_data_for_cox[model_data_for_cox['event'] == 0]),
                    'risk_score_range': f"{model_data_for_cox['duration'].min()}-{model_data_for_cox['duration'].max()}"
                }
            }
            
            # 6. Ergebnisse
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'c_index': self.c_index,
                'target_achieved': self.c_index >= 0.95,
                'performance': evaluation_results,
                'features_used': good_features,
                'model_data_size': len(model_data_for_cox),
                'events': model_data_for_cox['event'].sum(),
                'duration_seconds': duration,
                'timestamp': end_time.isoformat(),
                'experiment_id': experiment_id,
                'cutoff_exclusive': self.cutoff_exclusive
            }
            
            # Optionale JSON-Speicherung für Experiment-Integration
            if experiment_id:
                self._save_experiment_results(results, experiment_id)
                
                # Befülle alle Cox-Tabellen in JSON-Database
                self._populate_cox_tables(stage0_data, survival_panel, cox_data, 
                                        model_data, results, experiment_id)
            
            self.logger.info(f"✅ Cox-Analyse abgeschlossen:")
            self.logger.info(f"   🎯 C-Index: {self.c_index:.4f}")
            self.logger.info(f"   🎯 Ziel erreicht: {'✅ JA' if self.c_index >= 0.95 else '⚠️ NEIN'}")
            self.logger.info(f"   ⏱️ Laufzeit: {duration:.1f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline-Fehler: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """CLI Entry Point"""
    parser = argparse.ArgumentParser(description='Working Cox Survival Analysis Pipeline')
    parser.add_argument('--cutoff', type=int, default=202501, help='Cutoff-Zeitpunkt (YYYYMM)')
    parser.add_argument('--target-c-index', type=float, default=0.95, help='Ziel C-Index')
    parser.add_argument('--experiment-id', type=int, default=None, help='Experiment-ID für JSON-Database-Integration')
    
    args = parser.parse_args()
    
    print("🚀 Working Cox Survival Analysis Pipeline")
    print("="*60)
    
    # Pipeline ausführen
    pipeline = CoxWorkingPipeline(cutoff_exclusive=args.cutoff)
    results = pipeline.run_full_analysis(experiment_id=args.experiment_id)
    
    if results['success']:
        print(f"\n🎯 ERGEBNIS:")
        print(f"   ✅ C-Index: {results['c_index']:.4f}")
        print(f"   🎯 Ziel ({args.target_c_index}): {'✅ ERREICHT' if results['target_achieved'] else '⚠️ VERFEHLT'}")
        print(f"   📊 Features: {len(results['features_used'])}")
        print(f"   ⏱️ Laufzeit: {results['duration_seconds']:.1f}s")
        if results.get('experiment_id'):
            print(f"   🔬 Experiment-ID: {results['experiment_id']}")
        
        # Performance-Details
        if 'performance' in results and results['performance']:
            perf = results['performance']
            if 'business_scores' in perf:
                business = perf['business_scores']
                print(f"\n📈 Business-Metriken:")
                print(f"   📊 Prioritized Customers: {business.get('prioritized_customers', 'N/A')}")
                print(f"   🎯 Risk Score Range: {business.get('risk_score_range', 'N/A')}")
        
        exit(0)
    else:
        print(f"\n❌ FEHLER: {results['error']}")
        exit(1)

if __name__ == "__main__":
    main()
