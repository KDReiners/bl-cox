#!/usr/bin/env python3
"""
Cox Automatic Experiment Processor & Orchestrator
=================================================

Erweiterte Orchestrator-Funktionalität für Cox-Experimente:
- Automatische Parameter-Verwaltung aus Experiment-Hyperparametern
- Customer-Details-Generation nach Cox-Runs  
- Erweiterte Status-Management: 'created' → 'processing' → 'processed'/'failed'
- Reset-Funktionalität für fehlgeschlagene Experimente
- Batch-Verarbeitung mit cox_working_main.py Pipeline

Analog zum Churn-System Orchestrator-Pattern.

Features:
- Intelligente Cutoff-Parameter-Extraktion aus Experimenten
- Automatische Customer-Details-Generierung via SQL-Interface 
- Robuste Fehlerbehandlung und Status-Management
- Performance Monitoring über alle Cox-Experimente
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Füge Projekt-Root zum Python-Path hinzu
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bl.json_database.churn_json_database import ChurnJSONDatabase
from bl.Cox.cox_working_main import CoxWorkingPipeline

class CoxAutoProcessor:
    """
    Automatischer Cox-Experiment-Prozessor
    Verarbeitet alle unverarbeiteten Experimente aus der experiments Tabelle
    """
    
    def __init__(self, cutoff_exclusive: int = 202501):
        self.cutoff_exclusive = cutoff_exclusive
        self.logger = self._setup_logging()
        self.db = ChurnJSONDatabase()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('cox_auto_processor')

    def _progress(self, exp_id: int, phase: str, step: int, total: int, detail: str = "") -> None:
        try:
            self.logger.info(f"PROGRESS|exp_id={exp_id}|phase={phase}|step={step}|total={total}|detail={detail}")
        except Exception:
            pass
    
    def get_unprocessed_experiments(self) -> List[Dict[str, Any]]:
        """
        Ermittelt alle Experimente, die noch nicht verarbeitet wurden
        """
        self.logger.info("🔍 Suche nach unverarbeiteten Experimenten")
        
        # Alle Cox-Experimente mit Status 'created'
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        cox_experiments = [
            exp for exp in experiments 
            if exp.get("model_type", "").startswith("cox") 
            and exp.get("status", "") == "created"
        ]
        
        # Prüfe, welche bereits Cox-Daten haben
        cox_survival_exp_ids = set()
        cox_survival_records = self.db.data.get("tables", {}).get("cox_survival", {}).get("records", [])
        for record in cox_survival_records:
            cox_survival_exp_ids.add(record.get("id_experiments"))
        
        # Filtere unverarbeitete Experimente
        unprocessed = []
        for exp in cox_experiments:
            exp_id = exp.get("experiment_id")
            has_cox_data = exp_id in cox_survival_exp_ids
            
            if not has_cox_data:
                unprocessed.append(exp)
                self.logger.info(f"   📋 Unverarbeitet: ID {exp_id} - {exp.get('experiment_name', 'N/A')}")
            else:
                self.logger.info(f"   ✅ Bereits verarbeitet: ID {exp_id}")
        
        self.logger.info(f"🎯 Gefunden: {len(unprocessed)} unverarbeitete Experimente")
        return unprocessed
    
    def process_experiment(self, experiment: Dict[str, Any]) -> bool:
        """
        Verarbeitet ein einzelnes Experiment mit der Cox-Pipeline
        Erweiterte Orchestrator-Funktionalität mit Parameter-Management
        """
        exp_id = experiment.get("experiment_id")
        exp_name = experiment.get("experiment_name", "Unknown")
        
        self.logger.info(f"🚀 Starte Verarbeitung: Experiment {exp_id}")
        self.logger.info(f"   📋 Name: {exp_name}")
        
        try:
            # 🔧 ORCHESTRATOR: Parameter aus Experiment-Hyperparametern extrahieren
            experiment_cutoff = self._extract_experiment_cutoff(experiment)
            self.logger.info(f"   📅 Cutoff: {experiment_cutoff} (aus Experiment-Hyperparametern)")
            self._progress(exp_id, 'start', 0, 3, 'init')
            
            # Ziel-Experiment = übergebene ID (ein Experiment, zwei Läufe)
            target_exp_id = exp_id

            # Status auf 'processing' für Ziel-Experiment setzen (ohne sofortiges Speichern)
            self._update_experiment_status(target_exp_id, "processing", save=False)
            self._progress(target_exp_id, 'status', 1, 3, 'processing')
            
            # Cox-Pipeline für dieses Experiment starten
            pipeline = CoxWorkingPipeline(cutoff_exclusive=experiment_cutoff)
            self._progress(exp_id, 'analysis', 0, 3, 'run_full_analysis')
            results = pipeline.run_full_analysis(experiment_id=target_exp_id)
            
            if results.get("success", False):
                self.logger.info(f"✅ Experiment {exp_id} erfolgreich verarbeitet")
                self.logger.info(f"   🎯 C-Index: {results.get('c_index', 'N/A'):.4f}")
                self.logger.info(f"   ⏱️ Laufzeit: {results.get('duration_seconds', 0):.1f}s")
                self._progress(exp_id, 'analysis', 1, 3, 'completed')
                
                # 🔄 JSON-DB neu laden, um konkurrierende Saves zu vermeiden (Pipeline hat bereits geschrieben)
                try:
                    self.db = ChurnJSONDatabase()
                    self.logger.info("🔄 JSON-Database nach Pipeline-Lauf neu geladen")
                except Exception:
                    pass

                # 🔧 ORCHESTRATOR: Automatische Customer-Details-Generation (ohne sofortiges Speichern)
                self._generate_customer_details(target_exp_id, results, save=False)
                self._progress(target_exp_id, 'persist', 2, 3, 'customer_details')
                
                # Status auf 'processed' setzen
                self._update_experiment_status(target_exp_id, "processed", save=True)
                self._progress(target_exp_id, 'status', 3, 3, 'processed')
                
                return True
            else:
                self.logger.error(f"❌ Experiment {exp_id} Verarbeitung fehlgeschlagen")
                self.logger.error(f"   🔥 Fehler: {results.get('error', 'Unknown error')}")
                
                # Status auf 'failed' setzen
                self._update_experiment_status(exp_id, "failed", save=True)
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Experiment {exp_id} Exception: {e}")
            
            # Status auf 'failed' setzen
            self._update_experiment_status(exp_id, "failed", save=True)
            
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_experiment_cutoff(self, experiment: Dict[str, Any]) -> int:
        """
        🔧 ORCHESTRATOR: Extrahiert cutoff_exclusive aus Experiment-Hyperparametern
        KEIN FALLBACK - Experiment muss expliziten cutoff_exclusive Parameter haben
        """
        # Prüfe Hyperparameter
        hyperparameters = experiment.get("hyperparameters", {})
        
        if not hyperparameters:
            raise ValueError(f"❌ Experiment {experiment.get('experiment_id')} hat keine Hyperparameters - cutoff_exclusive fehlt")
        
        if "cutoff_exclusive" not in hyperparameters:
            available_params = list(hyperparameters.keys())
            raise ValueError(f"❌ Experiment {experiment.get('experiment_id')} - cutoff_exclusive nicht gefunden in Hyperparameters: {available_params}")
        
        cutoff = hyperparameters["cutoff_exclusive"]
        if not isinstance(cutoff, (int, str)) or not str(cutoff).isdigit():
            raise ValueError(f"❌ Experiment {experiment.get('experiment_id')} - ungültiger cutoff_exclusive Wert: '{cutoff}' (muss YYYYMM Format sein)")
            
        cutoff_int = int(cutoff)
        if cutoff_int < 200000 or cutoff_int > 999999:
            raise ValueError(f"❌ Experiment {experiment.get('experiment_id')} - cutoff_exclusive '{cutoff_int}' ist kein gültiges YYYYMM Format")
        
        self.logger.info(f"   ✅ Cutoff aus Experiment-Hyperparametern: {cutoff_int}")
        return cutoff_int
    
    def _generate_customer_details(self, experiment_id: int, results: Dict[str, Any], save: bool = True) -> None:
        """
        🔧 ORCHESTRATOR: Automatische Customer-Details-Generation nach Cox-Run
        
        Generiert erweiterte Customer-Details mit Survival-Wahrscheinlichkeiten
        und Risk-Profiling über das SQL-Interface
        """
        try:
            self.logger.info(f"📊 Generiere Customer-Details für Experiment {experiment_id}")
            
            # Prüfe ob Cox-Daten verfügbar sind
            cox_survival_records = self.db.data.get("tables", {}).get("cox_survival", {}).get("records", [])
            experiment_records = [r for r in cox_survival_records if r.get("id_experiments") == experiment_id]
            
            if not experiment_records:
                self.logger.warning(f"⚠️ Keine Cox-Survival-Daten für Experiment {experiment_id} gefunden")
                return
            
            self.logger.info(f"   📊 Verarbeite {len(experiment_records)} Customer-Records")
            
            # Customer-Details basierend auf Cox-Daten generieren
            customer_details = []
            for record in experiment_records:
                customer_detail = {
                    "Kunde": record.get("Kunde"),
                    "experiment_id": experiment_id,
                    "cox_analysis_type": "enhanced_survival_analysis",
                    "survival_months": round(record.get("duration", 0.0), 2),
                    "event_occurred": record.get("event", 0),
                    "customer_status": "Churned" if record.get("event") == 1 else "Active",
                    "c_index": results.get("c_index", 0.0),
                    "analysis_date": results.get("timestamp", ""),
                    "feature_count": len(results.get("features_used", [])),
                    "model_performance": "Excellent" if results.get("c_index", 0) > 0.8 else "Good",
                    "source": "cox"
                }
                customer_details.append(customer_detail)
            
            # Customer-Details zu customer_cox_details Tabelle hinzufügen
            success = self.db.add_customer_cox_details(customer_details)
            
            # Speichern (optional)
            if save:
                self.db.save()
            
            self.logger.info(f"   ✅ {len(customer_details)} Customer-Details generiert und gespeichert")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Customer-Details-Generation fehlgeschlagen: {e}")
    
    def reset_failed_experiments(self) -> int:
        """
        🔧 ORCHESTRATOR: Reset fehlgeschlagener Experimente
        
        Setzt den Status von 'failed' auf 'created' zurück für erneute Verarbeitung
        """
        self.logger.info("🔄 Suche nach fehlgeschlagenen Experimenten zum Reset")
        
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        failed_experiments = [
            exp for exp in experiments 
            if exp.get("model_type", "").startswith("cox") 
            and exp.get("status", "") == "failed"
        ]
        
        if not failed_experiments:
            self.logger.info("ℹ️ Keine fehlgeschlagenen Cox-Experimente gefunden")
            return 0
        
        reset_count = 0
        for exp in failed_experiments:
            exp_id = exp.get("experiment_id")
            exp_name = exp.get("experiment_name", "Unknown")
            
            try:
                # Status zurücksetzen
                exp["status"] = "created" 
                exp["reset_at"] = datetime.now().isoformat()
                exp["reset_reason"] = "Manual reset via orchestrator"
                
                self.logger.info(f"   ✅ Reset: Experiment {exp_id} - {exp_name}")
                reset_count += 1
                
            except Exception as e:
                self.logger.error(f"❌ Reset-Fehler für Experiment {exp_id}: {e}")
        
        if reset_count > 0:
            try:
                self.db.save()
                self.logger.info(f"✅ {reset_count} Experimente erfolgreich zurückgesetzt")
            except Exception as e:
                self.logger.error(f"❌ Fehler beim Speichern: {e}")
                return 0
        
        return reset_count
    
    def check_cutoff_parameters(self) -> Dict[str, Any]:
        """
        🔧 DIAGNOSTIK: Prüft cutoff_exclusive Parameter aller Cox-Experimente
        """
        self.logger.info("🔍 Prüfe cutoff_exclusive Parameter aller Cox-Experimente")
        
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        cox_experiments = [exp for exp in experiments if exp.get("model_type", "").startswith("cox")]
        
        results = {
            "total_cox_experiments": len(cox_experiments),
            "valid_cutoff": [],
            "missing_hyperparameters": [],
            "missing_cutoff_exclusive": [],
            "invalid_cutoff_format": [],
            "valid_experiments": 0,
            "problematic_experiments": 0
        }
        
        for exp in cox_experiments:
            exp_id = exp.get("experiment_id")
            exp_name = exp.get("experiment_name", "Unknown")
            
            try:
                # Verwende die normale Cutoff-Extraktion (ohne Fallback)
                cutoff = self._extract_experiment_cutoff(exp)
                results["valid_cutoff"].append({
                    "experiment_id": exp_id,
                    "experiment_name": exp_name,
                    "cutoff_exclusive": cutoff
                })
                results["valid_experiments"] += 1
                self.logger.info(f"   ✅ ID {exp_id}: cutoff_exclusive = {cutoff}")
                
            except ValueError as e:
                error_msg = str(e)
                results["problematic_experiments"] += 1
                
                if "hat keine Hyperparameters" in error_msg:
                    results["missing_hyperparameters"].append({
                        "experiment_id": exp_id,
                        "experiment_name": exp_name,
                        "error": error_msg
                    })
                    self.logger.warning(f"   ❌ ID {exp_id}: Keine Hyperparameters")
                    
                elif "cutoff_exclusive nicht gefunden" in error_msg:
                    hyperparams = list(exp.get("hyperparameters", {}).keys())
                    results["missing_cutoff_exclusive"].append({
                        "experiment_id": exp_id,
                        "experiment_name": exp_name,
                        "available_parameters": hyperparams,
                        "error": error_msg
                    })
                    self.logger.warning(f"   ❌ ID {exp_id}: cutoff_exclusive fehlt, verfügbar: {hyperparams}")
                    
                else:
                    results["invalid_cutoff_format"].append({
                        "experiment_id": exp_id,
                        "experiment_name": exp_name,
                        "error": error_msg
                    })
                    self.logger.warning(f"   ❌ ID {exp_id}: Ungültiges cutoff_exclusive Format")
                    
            except Exception as e:
                self.logger.error(f"   ❌ ID {exp_id}: Unerwarteter Fehler: {e}")
                results["problematic_experiments"] += 1
        
        return results
    
    def _update_experiment_status(self, experiment_id: int, status: str, save: bool = True) -> None:
        """
        Aktualisiert den Status eines Experiments in der Datenbank
        """
        try:
            self.logger.info(f"📊 Aktualisiere Status von Experiment {experiment_id} auf '{status}'")
            
            # Experiment in der Tabelle finden und Status aktualisieren
            experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
            
            for exp in experiments:
                if exp.get("experiment_id") == experiment_id:
                    exp["status"] = status
                    exp["processed_at"] = datetime.now().isoformat()
                    break
            
            # Datenbank speichern
            if save:
                self.db.save()
            self.logger.info(f"✅ Status aktualisiert: Experiment {experiment_id} → {status}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Status-Update fehlgeschlagen für Experiment {experiment_id}: {e}")
    
    def process_all_unprocessed(self) -> Dict[str, Any]:
        """
        Verarbeitet alle unverarbeiteten Experimente automatisch
        """
        self.logger.info("🚀 STARTE AUTOMATISCHE EXPERIMENT-VERARBEITUNG")
        self.logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # Unverarbeitete Experimente finden
        unprocessed_experiments = self.get_unprocessed_experiments()
        
        if not unprocessed_experiments:
            self.logger.info("✅ Keine unverarbeiteten Experimente gefunden")
            return {
                "success": True,
                "processed_count": 0,
                "failed_count": 0,
                "total_duration": 0,
                "message": "Alle Experimente bereits verarbeitet"
            }
        
        # Alle unverarbeiteten Experimente verarbeiten
        processed_count = 0
        failed_count = 0
        
        for i, experiment in enumerate(unprocessed_experiments, 1):
            exp_id = experiment.get("experiment_id")
            self.logger.info(f"\n📋 EXPERIMENT {i}/{len(unprocessed_experiments)}: ID {exp_id}")
            self.logger.info("-" * 50)
            
            success = self.process_experiment(experiment)
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
        
        # Zusammenfassung
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🎯 AUTOMATISCHE VERARBEITUNG ABGESCHLOSSEN")
        self.logger.info(f"✅ Erfolgreich verarbeitet: {processed_count}")
        self.logger.info(f"❌ Fehlgeschlagen: {failed_count}")
        self.logger.info(f"⏱️ Gesamtlaufzeit: {total_duration:.1f}s")
        
        return {
            "success": failed_count == 0,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_duration": total_duration,
            "experiments_processed": [exp.get("experiment_id") for exp in unprocessed_experiments]
        }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Gibt eine Übersicht über den Verarbeitungsstatus aller Experimente
        """
        experiments = self.db.data.get("tables", {}).get("experiments", {}).get("records", [])
        cox_experiments = [exp for exp in experiments if exp.get("model_type", "").startswith("cox")]
        
        status_summary = {}
        for exp in cox_experiments:
            status = exp.get("status", "unknown")
            status_summary[status] = status_summary.get(status, 0) + 1
        
        # Prüfe Cox-Daten-Verfügbarkeit
        cox_survival_exp_ids = set()
        cox_survival_records = self.db.data.get("tables", {}).get("cox_survival", {}).get("records", [])
        for record in cox_survival_records:
            cox_survival_exp_ids.add(record.get("id_experiments"))
        
        return {
            "total_cox_experiments": len(cox_experiments),
            "status_breakdown": status_summary,
            "experiments_with_cox_data": len(cox_survival_exp_ids),
            "experiments": [
                {
                    "experiment_id": exp.get("experiment_id"),
                    "experiment_name": exp.get("experiment_name"),
                    "status": exp.get("status"),
                    "has_cox_data": exp.get("experiment_id") in cox_survival_exp_ids,
                    "created_at": exp.get("created_at"),
                    "processed_at": exp.get("processed_at")
                }
                for exp in cox_experiments
            ]
        }

def main():
    """CLI Entry Point für Cox Orchestrator"""
    parser = argparse.ArgumentParser(description='Cox Automatic Experiment Processor & Orchestrator')
    parser.add_argument('--cutoff', type=int, default=202501, help='Cutoff-Zeitpunkt (YYYYMM)')
    parser.add_argument('--status', action='store_true', help='Zeige nur Status-Übersicht')
    parser.add_argument('--check-cutoff', action='store_true', help='🔧 Prüfe cutoff_exclusive Parameter aller Cox-Experimente')
    parser.add_argument('--reset-failed', action='store_true', help='🔧 ORCHESTRATOR: Reset fehlgeschlagener Experimente')
    parser.add_argument('--force', action='store_true', help='Erzwinge Verarbeitung auch bereits verarbeiteter Experimente')
    
    args = parser.parse_args()
    
    print("🔄 Cox Automatic Experiment Processor & Orchestrator")
    print("="*70)
    
    processor = CoxAutoProcessor(cutoff_exclusive=args.cutoff)
    
    if args.status:
        # Nur Status anzeigen
        status = processor.get_processing_status()
        
        print(f"\n📊 VERARBEITUNGSSTATUS:")
        print(f"   🎯 Cox-Experimente gesamt: {status['total_cox_experiments']}")
        print(f"   📈 Mit Cox-Daten: {status['experiments_with_cox_data']}")
        
        print(f"\n📋 STATUS-VERTEILUNG:")
        for status_name, count in status['status_breakdown'].items():
            print(f"   {status_name}: {count}")
        
        print(f"\n🔍 EXPERIMENT-DETAILS:")
        for exp in status['experiments']:
            exp_id = exp['experiment_id']
            name = exp['experiment_name'][:40] + "..." if len(exp['experiment_name']) > 40 else exp['experiment_name']
            status_str = exp['status']
            cox_data = "✅" if exp['has_cox_data'] else "❌"
            print(f"   ID {exp_id}: {name} [{status_str}] Cox-Data: {cox_data}")
        
    elif args.check_cutoff:
        # 🔧 DIAGNOSTIK: cutoff_exclusive Parameter prüfen
        print(f"\n🔧 CUTOFF-PARAMETER DIAGNOSE")
        print("=" * 50)
        
        check_results = processor.check_cutoff_parameters()
        
        print(f"\n📊 ÜBERSICHT:")
        print(f"   🎯 Cox-Experimente gesamt: {check_results['total_cox_experiments']}")
        print(f"   ✅ Gültige cutoff_exclusive: {check_results['valid_experiments']}")
        print(f"   ❌ Problematische: {check_results['problematic_experiments']}")
        
        if check_results['valid_cutoff']:
            print(f"\n✅ GÜLTIGE EXPERIMENTE:")
            for exp in check_results['valid_cutoff']:
                print(f"   ID {exp['experiment_id']}: cutoff_exclusive = {exp['cutoff_exclusive']}")
        
        if check_results['missing_hyperparameters']:
            print(f"\n❌ EXPERIMENTE OHNE HYPERPARAMETERS:")
            for exp in check_results['missing_hyperparameters']:
                print(f"   ID {exp['experiment_id']}: {exp['experiment_name'][:50]}...")
        
        if check_results['missing_cutoff_exclusive']:
            print(f"\n❌ EXPERIMENTE OHNE cutoff_exclusive:")
            for exp in check_results['missing_cutoff_exclusive']:
                params = exp['available_parameters']
                print(f"   ID {exp['experiment_id']}: verfügbare Parameter: {params}")
        
        if check_results['invalid_cutoff_format']:
            print(f"\n❌ EXPERIMENTE MIT UNGÜLTIGEM cutoff_exclusive:")
            for exp in check_results['invalid_cutoff_format']:
                print(f"   ID {exp['experiment_id']}: {exp['error']}")
        
        if check_results['problematic_experiments'] > 0:
            print(f"\n💡 LÖSUNGSVORSCHLAG:")
            print(f"   Füge cutoff_exclusive zu Experiment-Hyperparametern hinzu:")
            print(f"   'hyperparameters': {{'cutoff_exclusive': 202501, ...}}")
        
    elif args.reset_failed:
        # 🔧 ORCHESTRATOR: Reset fehlgeschlagener Experimente
        print(f"\n🔧 ORCHESTRATOR: RESET FEHLGESCHLAGENER EXPERIMENTE")
        print("=" * 50)
        
        reset_count = processor.reset_failed_experiments()
        
        if reset_count > 0:
            print(f"\n✅ RESET ERFOLGREICH!")
            print(f"   🔄 {reset_count} Experimente zurückgesetzt")
            print(f"   📋 Status: 'failed' → 'created'")
            print(f"\n💡 Führe jetzt den Orchestrator aus, um die Experimente zu verarbeiten:")
            print(f"   python bl/Cox/cox_auto_processor.py --cutoff {args.cutoff}")
        else:
            print(f"\n ℹ️ KEINE EXPERIMENTE ZUM RESET")
            print(f"   📊 Alle Cox-Experimente sind in ordnungsgemäßem Status")
        
    else:
        # Automatische Verarbeitung starten
        results = processor.process_all_unprocessed()
        
        if results['success']:
            print(f"\n🎉 ORCHESTRATOR ERFOLGREICH ABGESCHLOSSEN!")
            print(f"   ✅ Verarbeitet: {results['processed_count']} Cox-Experimente")
            print(f"   ⏱️ Laufzeit: {results['total_duration']:.1f}s")
            print(f"   📊 Customer-Details: Automatisch generiert")
            print(f"   🔗 SQL-Integration: Verfügbar über cox_survival_enhanced")
            exit(0)
        else:
            print(f"\n⚠️ ORCHESTRATOR TEILWEISE ERFOLGREICH")
            print(f"   ✅ Verarbeitet: {results['processed_count']}")
            print(f"   ❌ Fehlgeschlagen: {results['failed_count']}")
            print(f"   ⏱️ Laufzeit: {results['total_duration']:.1f}s")
            print(f"\n💡 Tipp: Verwende --reset-failed um fehlgeschlagene Experimente zurückzusetzen")
            exit(1)

if __name__ == "__main__":
    main()
