#!/usr/bin/env python3
"""
Cox Main - Clean Orchestrator für Cox-Survival-Analyse
======================================================

Produktionsbereiter Orchestrator für die neue Cox-Pipeline-Architektur.

Autor: AI Assistant
Datum: 2025-01-27
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Core Module Imports
try:
    from .cox_data_loader import CoxDataLoader
    from .cox_feature_engine import CoxFeatureEngine  
    from .cox_model_trainer import CoxModelTrainer
    from .cox_evaluator import CoxEvaluator
except ImportError:
    from cox_data_loader import CoxDataLoader
    from cox_feature_engine import CoxFeatureEngine
    from cox_model_trainer import CoxModelTrainer
    from cox_evaluator import CoxEvaluator


class CoxPipeline:
    """Clean Orchestrator für Cox-Survival-Analyse Pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        
        # Core-Module initialisieren
        self.data_loader = CoxDataLoader(cutoff_exclusive=self.config['cutoff_exclusive'])
        self.feature_engine = CoxFeatureEngine(feature_config=self.config.get('feature_config'))
        self.model_trainer = CoxModelTrainer(model_config=self.config.get('model_config'))
        self.evaluator = CoxEvaluator(evaluation_config=self.config.get('evaluation_config'))
        
        self.logger.info("🚀 Cox Pipeline initialisiert")
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'cutoff_exclusive': 202501,
            'target_c_index': 0.95,
            'enable_hyperparameter_tuning': True,
            'save_intermediate_results': True,
            'feature_config': {'rolling_windows': [6, 12, 18], 'feature_selection_k': 15},
            'model_config': {'penalizer': 0.01, 'target_c_index': 0.95},
            'evaluation_config': {'prioritization': {'time_horizons': [6, 12]}}
        }
    
    def run_full_analysis(self, churn_experiment_id: Optional[int] = None) -> Dict[str, Any]:
        """Führt komplette Cox-Analyse durch"""
        self.logger.info("🚀 Starte vollständige Cox-Analyse")
        start_time = datetime.now()
        
        try:
            # 1. Daten laden (OHNE Data Leakage)
            self.logger.info("📊 Lade Daten")
            stage0_data = self.data_loader.load_stage0_data()
            survival_panel = self.data_loader.create_survival_panel(stage0_data)
            alive_customers = self.data_loader.get_alive_customers_at_cutoff()
            
            # 2. Features erstellen
            self.logger.info("⚙️ Erstelle Features")
            features = self.feature_engine.create_cox_features(survival_panel, stage0_data)
            
            # 3. Modell trainieren
            self.logger.info("🎯 Trainiere Modell")
            training_data = self.model_trainer.prepare_cox_data(survival_panel, features)
            
            if self.config['enable_hyperparameter_tuning']:
                self.model_trainer.hyperparameter_tuning(training_data)
            
            model = self.model_trainer.train_cox_model(training_data)
            performance = self.model_trainer.evaluate_model_performance(model, training_data)
            
            # 4. Kunden priorisieren
            self.logger.info("🎯 Priorisiere Kunden")
            alive_customer_data = stage0_data[stage0_data['Kunde'].isin(alive_customers)].copy()
            
            prioritization_data = self.evaluator.generate_customer_prioritization(
                model=model,
                alive_customers_data=alive_customer_data,
                cutoff_month=self.config['cutoff_exclusive']
            )
            
            # 5. Ergebnisse speichern
            prioritization_path = self.evaluator.save_prioritization_results(prioritization_data)
            
            # Finale Zusammenfassung
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            c_index = performance['concordance_index']
            target_achieved = c_index >= self.config['target_c_index']
            
            results = {
                'metadata': {
                    'timestamp': end_time.isoformat(),
                    'execution_time_seconds': execution_time,
                    'churn_experiment_id': churn_experiment_id
                },
                'performance_summary': {
                    'c_index_achieved': c_index,
                    'target_achieved': target_achieved,
                    'vs_historical': c_index / 0.993,
                    'vs_optimized': c_index / 0.890
                },
                'data_summary': {
                    'customers_analyzed': len(stage0_data['Kunde'].unique()),
                    'survival_records': len(survival_panel),
                    'features_created': len(features.columns) - 1,
                    'customers_prioritized': len(prioritization_data)
                },
                'output_files': {
                    'prioritization': str(prioritization_path)
                }
            }
            
            self.logger.info(f"✅ Analyse abgeschlossen - C-Index: {c_index:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline-Fehler: {e}")
            raise
    
    def run_prioritization_only(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """Führt nur Kunden-Priorisierung durch"""
        self.logger.info("🎯 Starte Priorisierung")
        
        try:
            # Modell laden
            if model_path:
                model = self.model_trainer.load_model(model_path)
            else:
                raise ValueError("Kein Modell-Pfad angegeben")
            
            # Daten laden
            stage0_data = self.data_loader.load_stage0_data()
            alive_customers = self.data_loader.get_alive_customers_at_cutoff()
            alive_customer_data = stage0_data[stage0_data['Kunde'].isin(alive_customers)].copy()
            
            # Priorisierung durchführen
            prioritization_data = self.evaluator.generate_customer_prioritization(
                model=model,
                alive_customers_data=alive_customer_data,
                cutoff_month=self.config['cutoff_exclusive']
            )
            
            # Speichern
            output_path = self.evaluator.save_prioritization_results(prioritization_data)
            
            results = {
                'success': True,
                'customers_prioritized': len(prioritization_data),
                'avg_priority_score': prioritization_data['PriorityScore'].mean(),
                'high_risk_customers': len(prioritization_data[prioritization_data['PriorityScore'] >= 70]),
                'output_path': str(output_path)
            }
            
            self.logger.info(f"✅ Priorisierung abgeschlossen: {len(prioritization_data)} Kunden")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Priorisierung-Fehler: {e}")
            return {'success': False, 'error': str(e)}


def create_argument_parser() -> argparse.ArgumentParser:
    """Erstellt Command-Line-Interface"""
    parser = argparse.ArgumentParser(description='Cox Survival Analysis Pipeline v2.0')
    
    # Hauptmodi
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full-analysis', action='store_true', help='Vollständige Cox-Analyse')
    mode_group.add_argument('--prioritization-only', action='store_true', help='Nur Kunden-Priorisierung')
    
    # Parameter
    parser.add_argument('--cutoff', type=int, default=202501, help='Cutoff-Zeitpunkt (YYYYMM)')
    parser.add_argument('--target-c-index', type=float, default=0.95, help='Ziel-C-Index')
    parser.add_argument('--churn-experiment', type=int, help='Churn-Experiment-ID')
    
    # Optionen
    parser.add_argument('--tune-hyperparameters', action='store_true', help='Hyperparameter-Tuning aktivieren')
    parser.add_argument('--model', type=Path, help='Pfad zu gespeichertem Modell (für --prioritization-only)')
    parser.add_argument('--config', type=Path, help='Pfad zu Konfigurationsdatei (JSON)')
    
    return parser


def main():
    """Hauptfunktion für Command-Line-Ausführung"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Logging konfigurieren
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Cox Survival Analysis Pipeline v2.0")
    logger.info("=" * 50)
    
    try:
        # Konfiguration laden
        config = {}
        if args.config and args.config.exists():
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # CLI-Argumente in Config integrieren
        config.update({
            'cutoff_exclusive': args.cutoff,
            'target_c_index': args.target_c_index,
            'enable_hyperparameter_tuning': args.tune_hyperparameters
        })
        
        # Pipeline initialisieren
        pipeline = CoxPipeline(config=config)
        
        # Aktion ausführen
        if args.full_analysis:
            logger.info("🎯 Modus: Vollständige Analyse")
            results = pipeline.run_full_analysis(churn_experiment_id=args.churn_experiment)
            
            performance = results['performance_summary']
            print("\n" + "=" * 60)
            print("🎉 VOLLSTÄNDIGE ANALYSE ERFOLGREICH ABGESCHLOSSEN")
            print("=" * 60)
            print(f"🎯 C-Index erreicht: {performance['c_index_achieved']:.4f}")
            print(f"✅ Ziel erreicht: {'Ja' if performance['target_achieved'] else 'Nein'}")
            print(f"📈 vs. Historical (0.993): {performance['vs_historical']:.1%}")
            print(f"📈 vs. Optimized (0.890): {performance['vs_optimized']:.1%}")
            print(f"⏱️ Ausführungszeit: {results['metadata']['execution_time_seconds']:.2f}s")
            print(f"👥 Kunden analysiert: {results['data_summary']['customers_analyzed']}")
            print(f"📊 Features erstellt: {results['data_summary']['features_created']}")
            print(f"🎯 Kunden priorisiert: {results['data_summary']['customers_prioritized']}")
            print(f"💾 Priorisierung: {results['output_files']['prioritization']}")
            print("=" * 60)
        
        elif args.prioritization_only:
            logger.info("🎯 Modus: Nur Priorisierung")
            results = pipeline.run_prioritization_only(model_path=args.model)
            
            if results['success']:
                print("\n" + "=" * 50)
                print("🎯 PRIORISIERUNG ERFOLGREICH ABGESCHLOSSEN")
                print("=" * 50)
                print(f"👥 Kunden priorisiert: {results['customers_prioritized']}")
                print(f"📊 Ø Priority Score: {results['avg_priority_score']:.1f}")
                print(f"🔴 High-Risk Kunden: {results['high_risk_customers']}")
                print(f"💾 Output: {results['output_path']}")
                print("=" * 50)
            else:
                print(f"❌ Priorisierung fehlgeschlagen: {results['error']}")
                sys.exit(1)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline durch Benutzer unterbrochen")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()