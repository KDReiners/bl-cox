#!/usr/bin/env python3
"""
Cox Evaluator - Performance-Evaluation und Priorisierung
=========================================================

Business-Interface für Cox-Modell-Ergebnisse mit Fokus auf:
- Performance-Evaluation und Metriken
- Kunden-Priorisierung mit Priority Score (0-100)
- Survival-Analyse und Visualisierungen  
- Business-Reports und KPI-Tracking

Konsolidiert Funktionalität aus:
- cox_analyzer.py (Analyse-Funktionen)
- cox_priorization.py (Priorisierung)

Autor: AI Assistant
Datum: 2025-01-27
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import logging
import sys
import warnings

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ matplotlib/seaborn nicht verfügbar - Visualisierungen deaktiviert")
    MATPLOTLIB_AVAILABLE = False

# Lifelines imports
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    print("⚠️ lifelines nicht verfügbar - Cox-Funktionen limitiert")
    LIFELINES_AVAILABLE = False

# Projekt-Pfade hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.paths_config import ProjectPaths
    from bl.Cox.cox_constants import (
        HUNDRED_PERCENT, SIX_MONTH_HORIZON, TWELVE_MONTH_HORIZON,
        FALSE_VALUE, TRUE_VALUE
    )
except ImportError as e:
    print(f"⚠️ Import-Fehler: {e}")
    # Fallback-Konstanten
    HUNDRED_PERCENT = 100
    SIX_MONTH_HORIZON = 6
    TWELVE_MONTH_HORIZON = 12
    FALSE_VALUE = 0
    TRUE_VALUE = 1


class CoxEvaluator:
    """
    Performance-Evaluation und Kunden-Priorisierung für Cox-Modelle
    """
    
    def __init__(self, evaluation_config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert Evaluator
        
        Args:
            evaluation_config: Konfiguration für Evaluation
        """
        self.config = evaluation_config or self._default_config()
        self.logger = self._setup_logging()
        
        # Paths
        try:
            self.paths = ProjectPaths()
            self.output_dir = self.paths.dynamic_outputs_directory() / "cox_analysis"
            self.prioritization_dir = self.paths.dynamic_outputs_directory() / "prioritization"
            self.visualization_dir = self.output_dir / "visualizations"
        except:
            self.output_dir = Path("dynamic_system_outputs/cox_analysis")
            self.prioritization_dir = Path("dynamic_system_outputs/prioritization")
            self.visualization_dir = self.output_dir / "visualizations"
        
        # Erstelle Output-Verzeichnisse
        self.output_dir.mkdir(exist_ok=True)
        self.prioritization_dir.mkdir(exist_ok=True)
        if MATPLOTLIB_AVAILABLE:
            self.visualization_dir.mkdir(exist_ok=True)
        
        # State
        self.evaluation_results: Dict[str, Any] = {}
        self.prioritization_results: Optional[pd.DataFrame] = None
        self.survival_curves: Dict[str, Any] = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        
        self.logger.info("📈 Cox Evaluator initialisiert")
    
    def _setup_logging(self) -> logging.Logger:
        """Konfiguriert Logging"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _default_config(self) -> Dict[str, Any]:
        """Standard-Konfiguration für Evaluation"""
        return {
            # Priorisierung
            'prioritization': {
                'time_horizons': [SIX_MONTH_HORIZON, TWELVE_MONTH_HORIZON],  # 6, 12 Monate
                'score_weights': {
                    'p6_weight': 0.3,      # Gewicht 6-Monats-Risiko
                    'p12_weight': 0.5,     # Gewicht 12-Monats-Risiko
                    'mtl_weight': 0.2      # Gewicht Months-to-Live
                },
                'rmst_horizons': [12, 24],  # RMST für 12 und 24 Monate
                'priority_score_range': [0, 100]  # Priority Score 0-100
            },
            
            # Visualisierung
            'visualization': {
                'save_plots': True,
                'plot_format': 'png',
                'plot_dpi': 300,
                'figsize': (12, 8),
                'style': 'seaborn-v0_8',
                'color_palette': 'viridis'
            },
            
            # Reporting
            'reporting': {
                'include_feature_importance': True,
                'include_survival_curves': True,
                'include_business_metrics': True,
                'decimal_places': 4
            },
            
            # Performance-Metriken
            'performance': {
                'time_points': [6, 12, 24, 36],  # Monate für time-dependent metrics
                'calibration_bins': 10,
                'bootstrap_samples': 100  # Für Konfidenz-Intervalle
            }
        }
    
    # =============================================================================
    # PERFORMANCE EVALUATION
    # =============================================================================
    
    def evaluate_model_performance(self, model: CoxPHFitter, 
                                 test_data: pd.DataFrame,
                                 duration_col: str = 'duration',
                                 event_col: str = 'event') -> Dict[str, Any]:
        """
        Umfassende Model-Performance-Evaluation
        
        Args:
            model: Trainiertes Cox-Modell
            test_data: Test-Datensatz
            duration_col: Duration-Spalte
            event_col: Event-Spalte
            
        Returns:
            Performance-Metriken
        """
        self.logger.info("📊 Starte umfassende Performance-Evaluation")
        
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines erforderlich für Performance-Evaluation")
        
        performance_metrics = {}
        
        # 1. Basis-Metriken
        self.logger.info("   📊 Berechne Basis-Metriken")
        basic_metrics = self._calculate_basic_metrics(model, test_data, duration_col, event_col)
        performance_metrics['basic_metrics'] = basic_metrics
        
        # 2. Zeit-abhängige Metriken
        self.logger.info("   📅 Berechne zeit-abhängige Metriken")
        time_dependent_metrics = self.calculate_time_dependent_metrics(
            model, test_data, duration_col, event_col
        )
        performance_metrics['time_dependent'] = time_dependent_metrics
        
        # 3. Feature-Importance
        if self.config['reporting']['include_feature_importance']:
            self.logger.info("   🔍 Analysiere Feature-Importance")
            feature_importance = self._extract_feature_importance(model)
            performance_metrics['feature_importance'] = feature_importance
        
        # 4. Model-Diagnostics
        self.logger.info("   🔧 Prüfe Model-Diagnostics")
        model_diagnostics = self._calculate_model_diagnostics(model, test_data)
        performance_metrics['model_diagnostics'] = model_diagnostics
        
        # 5. Performance-Summary
        performance_summary = self._create_performance_summary(performance_metrics)
        performance_metrics['summary'] = performance_summary
        
        # Ergebnisse speichern
        self.evaluation_results['performance'] = performance_metrics
        
        self.logger.info(f"✅ Performance-Evaluation abgeschlossen:")
        self.logger.info(f"   🎯 C-Index: {basic_metrics['concordance_index']:.4f}")
        self.logger.info(f"   📊 AIC: {basic_metrics['aic']:.2f}")
        self.logger.info(f"   📊 Events: {basic_metrics['event_count']}")
        
        return performance_metrics
    
    def _calculate_basic_metrics(self, model: CoxPHFitter, data: pd.DataFrame,
                               duration_col: str, event_col: str) -> Dict[str, Any]:
        """Berechnet Basis-Performance-Metriken"""
        feature_cols = [col for col in data.columns if col not in [duration_col, event_col]]
        
        # Concordance Index
        predictions = model.predict_partial_hazard(data[feature_cols])
        c_index = concordance_index(data[duration_col], predictions, data[event_col])
        
        # Model-Statistiken
        basic_metrics = {
            'concordance_index': float(c_index),
            'log_likelihood': float(model.log_likelihood_),
            'aic': float(model.AIC_),
            'bic': float(model.AIC_ + (np.log(len(data)) - 2) * len(feature_cols)),
            'partial_aic': float(model.AIC_partial_),
            'number_of_observations': len(data),
            'number_of_events': int(data[event_col].sum()),
            'event_rate': float(data[event_col].mean()),
            'number_of_features': len(feature_cols),
            'median_duration': float(data[duration_col].median()),
            'max_duration': float(data[duration_col].max())
        }
        
        return basic_metrics
    
    def calculate_time_dependent_metrics(self, model: CoxPHFitter, data: pd.DataFrame,
                                       duration_col: str = 'duration', 
                                       event_col: str = 'event') -> Dict[str, Any]:
        """
        Berechnet zeit-abhängige Performance-Metriken
        
        Args:
            model: Cox-Modell
            data: Test-Daten
            duration_col: Duration-Spalte
            event_col: Event-Spalte
            
        Returns:
            Zeit-abhängige Metriken
        """
        time_points = self.config['performance']['time_points']
        feature_cols = [col for col in data.columns if col not in [duration_col, event_col]]
        
        time_metrics = {
            'time_points': time_points,
            'auc_scores': {},
            'sensitivity': {},
            'specificity': {},
            'precision': {},
            'recall': {}
        }
        
        # Vorhersagen für alle Zeitpunkte
        try:
            survival_probs = model.predict_survival_function(data[feature_cols])
            
            for time_point in time_points:
                if time_point <= data[duration_col].max():
                    # Survival-Wahrscheinlichkeit zum Zeitpunkt
                    surv_at_time = []
                    for i, sf in enumerate(survival_probs):
                        if time_point in sf.index:
                            surv_at_time.append(sf.loc[time_point])
                        else:
                            # Interpolation oder nächster Wert
                            available_times = sf.index[sf.index <= time_point]
                            if len(available_times) > 0:
                                surv_at_time.append(sf.loc[available_times[-1]])
                            else:
                                surv_at_time.append(1.0)  # Noch nicht erreicht
                    
                    # Churn-Wahrscheinlichkeit (1 - Survival)
                    churn_probs = 1 - np.array(surv_at_time)
                    
                    # True Labels zum Zeitpunkt
                    true_events = ((data[duration_col] <= time_point) & (data[event_col] == 1)).astype(int)
                    
                    # AUC berechnen (vereinfacht)
                    if len(np.unique(true_events)) > 1:  # Nur wenn beide Klassen vorhanden
                        try:
                            from sklearn.metrics import roc_auc_score, precision_recall_curve
                            auc = roc_auc_score(true_events, churn_probs)
                            time_metrics['auc_scores'][time_point] = float(auc)
                            
                            # Precision/Recall bei optimalem Threshold
                            precision, recall, thresholds = precision_recall_curve(true_events, churn_probs)
                            optimal_idx = np.argmax(precision + recall)
                            
                            time_metrics['precision'][time_point] = float(precision[optimal_idx])
                            time_metrics['recall'][time_point] = float(recall[optimal_idx])
                            
                        except ImportError:
                            # Fallback ohne sklearn
                            time_metrics['auc_scores'][time_point] = 0.5
                        except:
                            time_metrics['auc_scores'][time_point] = 0.5
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Zeit-abhängige Metriken Berechnung fehlgeschlagen: {e}")
            time_metrics['error'] = str(e)
        
        return time_metrics
    
    def _extract_feature_importance(self, model: CoxPHFitter) -> List[Dict[str, Any]]:
        """Extrahiert Feature-Importance aus Cox-Modell"""
        try:
            summary = model.summary
            
            feature_importance = []
            for feature in summary.index:
                importance = {
                    'feature': feature,
                    'coefficient': float(summary.loc[feature, 'coef']),
                    'hazard_ratio': float(summary.loc[feature, 'exp(coef)']),
                    'p_value': float(summary.loc[feature, 'p']),
                    'confidence_lower': float(summary.loc[feature, 'exp(coef) lower 95%']),
                    'confidence_upper': float(summary.loc[feature, 'exp(coef) upper 95%']),
                    'abs_coefficient': float(abs(summary.loc[feature, 'coef'])),
                    'significant': float(summary.loc[feature, 'p']) < 0.05
                }
                feature_importance.append(importance)
            
            # Nach absolutem Koeffizienten sortieren
            feature_importance.sort(key=lambda x: x['abs_coefficient'], reverse=True)
            
            # Als DataFrame für spätere Verwendung speichern
            self.feature_importance = pd.DataFrame(feature_importance)
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature-Importance Extraktion fehlgeschlagen: {e}")
            return []
    
    def _calculate_model_diagnostics(self, model: CoxPHFitter, data: pd.DataFrame) -> Dict[str, Any]:
        """Berechnet Model-Diagnostics"""
        diagnostics = {
            'convergence_achieved': True,
            'model_warnings': [],
            'data_quality_issues': []
        }
        
        try:
            # Prüfe Konvergenz
            if hasattr(model, 'standard_errors_'):
                max_se = model.standard_errors_.max()
                if max_se > 10:
                    diagnostics['convergence_achieved'] = False
                    diagnostics['model_warnings'].append(f"Hohe Standard-Errors: {max_se:.2f}")
            
            # Prüfe extreme Koeffizienten
            if hasattr(model, 'params_'):
                max_coef = np.abs(model.params_).max()
                if max_coef > 5:
                    diagnostics['model_warnings'].append(f"Extreme Koeffizienten: {max_coef:.2f}")
            
            # Prüfe Event-Rate
            event_rate = data['event'].mean()
            if event_rate < 0.05:
                diagnostics['data_quality_issues'].append(f"Niedrige Event-Rate: {event_rate:.3f}")
            elif event_rate > 0.8:
                diagnostics['data_quality_issues'].append(f"Hohe Event-Rate: {event_rate:.3f}")
            
            # Gesamt-Assessment
            diagnostics['overall_quality'] = 'good'
            if not diagnostics['convergence_achieved'] or len(diagnostics['model_warnings']) > 2:
                diagnostics['overall_quality'] = 'fair'
            if len(diagnostics['data_quality_issues']) > 0:
                diagnostics['overall_quality'] = 'poor' if diagnostics['overall_quality'] == 'fair' else 'fair'
            
        except Exception as e:
            diagnostics['error'] = str(e)
            diagnostics['overall_quality'] = 'unknown'
        
        return diagnostics
    
    def _create_performance_summary(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Performance-Summary"""
        basic = performance_metrics.get('basic_metrics', {})
        
        c_index = basic.get('concordance_index', 0.5)
        
        # Performance-Level bestimmen
        if c_index >= 0.9:
            performance_level = "Excellent"
        elif c_index >= 0.8:
            performance_level = "Good"
        elif c_index >= 0.7:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        summary = {
            'performance_level': performance_level,
            'c_index': c_index,
            'model_quality': performance_metrics.get('model_diagnostics', {}).get('overall_quality', 'unknown'),
            'event_rate': basic.get('event_rate', 0),
            'sample_size': basic.get('number_of_observations', 0),
            'feature_count': basic.get('number_of_features', 0),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    # =============================================================================
    # CUSTOMER PRIORITIZATION
    # =============================================================================
    
    def generate_customer_prioritization(self, model: CoxPHFitter,
                                       alive_customers_data: pd.DataFrame,
                                       cutoff_month: int = 202501) -> pd.DataFrame:
        """
        Generiert Kunden-Priorisierung basierend auf Cox-Modell
        
        Args:
            model: Trainiertes Cox-Modell
            alive_customers_data: Daten der lebenden Kunden
            cutoff_month: Cutoff-Zeitpunkt (YYYYMM)
            
        Returns:
            DataFrame mit Priorisierung
        """
        self.logger.info(f"🎯 Generiere Kunden-Priorisierung für {len(alive_customers_data)} Kunden")
        
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines erforderlich für Kunden-Priorisierung")
        
        # Feature-Spalten identifizieren
        feature_cols = [col for col in alive_customers_data.columns 
                       if col not in ['Kunde', 'I_TIMEBASE', 'I_Alive', 'duration', 'event']]
        
        if len(feature_cols) == 0:
            raise ValueError("Keine Feature-Spalten für Priorisierung gefunden")
        
        self.logger.info(f"   🔢 Verwende {len(feature_cols)} Features")
        
        prioritization_records = []
        time_horizons = self.config['prioritization']['time_horizons']
        rmst_horizons = self.config['prioritization']['rmst_horizons']
        
        # Pro Kunde Priorisierung berechnen
        for _, kunde_row in alive_customers_data.iterrows():
            kunde_id = kunde_row['Kunde']
            
            try:
                # Feature-Vektor für diesen Kunden
                customer_features = kunde_row[feature_cols].to_frame().T
                
                # Survival-Wahrscheinlichkeiten berechnen
                survival_probs = self._calculate_survival_probabilities(
                    model, customer_features, time_horizons
                )
                
                # RMST berechnen
                rmst_values = self._calculate_rmst(model, customer_features, rmst_horizons)
                
                # Months-to-Live schätzen
                months_to_live = self._calculate_months_to_live(model, customer_features)
                
                # Priority Score berechnen
                priority_score = self._calculate_priority_score(
                    survival_probs, months_to_live
                )
                
                # Priorisierungs-Record erstellen
                record = {
                    'Kunde': kunde_id,
                    'P_Event_6m': 1 - survival_probs.get(6, 1.0),  # Churn-Wahrscheinlichkeit
                    'P_Event_12m': 1 - survival_probs.get(12, 1.0),
                    'RMST_12m': rmst_values.get(12, 12.0),
                    'RMST_24m': rmst_values.get(24, 24.0),
                    'MonthsToLive_Conditional': months_to_live,
                    'PriorityScore': priority_score,
                    'RiskLevel': self._categorize_risk_level(priority_score),
                    'ProcessingTimestamp': datetime.now().isoformat()
                }
                
                prioritization_records.append(record)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Priorisierung für Kunde {kunde_id} fehlgeschlagen: {e}")
                continue
        
        if not prioritization_records:
            raise ValueError("Keine Priorisierung-Records erstellt")
        
        # DataFrame erstellen
        prioritization_df = pd.DataFrame(prioritization_records)
        
        # Nach Priority Score sortieren (höchstes Risiko zuerst)
        prioritization_df = prioritization_df.sort_values('PriorityScore', ascending=False)
        
        # Priorisierungs-Statistiken
        self._log_prioritization_stats(prioritization_df)
        
        # Ergebnisse speichern
        self.prioritization_results = prioritization_df
        
        self.logger.info(f"✅ Kunden-Priorisierung abgeschlossen: {len(prioritization_df)} Kunden")
        
        return prioritization_df
    
    def _calculate_survival_probabilities(self, model: CoxPHFitter, customer_features: pd.DataFrame,
                                        time_horizons: List[int]) -> Dict[int, float]:
        """Berechnet Survival-Wahrscheinlichkeiten für Zeiträume"""
        survival_probs = {}
        
        try:
            # Survival-Funktion für Kunden
            survival_function = model.predict_survival_function(customer_features)
            sf = survival_function.iloc[:, 0]  # Erste (einzige) Spalte
            
            for horizon in time_horizons:
                if horizon in sf.index:
                    survival_probs[horizon] = float(sf.loc[horizon])
                else:
                    # Interpolation oder nächster Wert
                    available_times = sf.index[sf.index <= horizon]
                    if len(available_times) > 0:
                        survival_probs[horizon] = float(sf.loc[available_times[-1]])
                    else:
                        survival_probs[horizon] = 1.0  # Noch nicht erreicht
            
        except Exception as e:
            self.logger.warning(f"⚠️ Survival-Wahrscheinlichkeiten Berechnung fehlgeschlagen: {e}")
            # Fallback-Werte
            for horizon in time_horizons:
                survival_probs[horizon] = 0.8  # Konservativer Fallback
        
        return survival_probs
    
    def _calculate_rmst(self, model: CoxPHFitter, customer_features: pd.DataFrame,
                       rmst_horizons: List[int]) -> Dict[int, float]:
        """Berechnet Restricted Mean Survival Time"""
        rmst_values = {}
        
        try:
            survival_function = model.predict_survival_function(customer_features)
            sf = survival_function.iloc[:, 0]
            
            for horizon in rmst_horizons:
                # RMST = Integral der Survival-Funktion bis horizon
                times_in_horizon = sf.index[sf.index <= horizon]
                if len(times_in_horizon) > 1:
                    # Trapezoid-Regel für numerische Integration
                    rmst = np.trapz(sf.loc[times_in_horizon], times_in_horizon)
                    rmst_values[horizon] = float(rmst)
                else:
                    rmst_values[horizon] = float(horizon * 0.8)  # Fallback
        
        except Exception as e:
            self.logger.warning(f"⚠️ RMST-Berechnung fehlgeschlagen: {e}")
            # Fallback-Werte
            for horizon in rmst_horizons:
                rmst_values[horizon] = float(horizon * 0.8)
        
        return rmst_values
    
    def _calculate_months_to_live(self, model: CoxPHFitter, customer_features: pd.DataFrame) -> float:
        """Berechnet erwartete Months-to-Live"""
        try:
            survival_function = model.predict_survival_function(customer_features)
            sf = survival_function.iloc[:, 0]
            
            # Median Survival Time (50% Survival-Wahrscheinlichkeit)
            median_survival = None
            for time, prob in sf.items():
                if prob <= 0.5:
                    median_survival = time
                    break
            
            if median_survival is not None:
                return float(median_survival)
            else:
                # Fallback: RMST für 36 Monate
                times_36m = sf.index[sf.index <= 36]
                if len(times_36m) > 1:
                    rmst_36 = np.trapz(sf.loc[times_36m], times_36m)
                    return float(rmst_36)
                else:
                    return 24.0  # Konservativer Fallback
        
        except Exception as e:
            self.logger.warning(f"⚠️ Months-to-Live Berechnung fehlgeschlagen: {e}")
            return 24.0  # Fallback
    
    def _calculate_priority_score(self, survival_probs: Dict[int, float], 
                                months_to_live: float) -> float:
        """
        Berechnet Priority Score (0-100, höher = risikanter)
        
        Args:
            survival_probs: Survival-Wahrscheinlichkeiten nach Zeiträumen
            months_to_live: Erwartete Months-to-Live
            
        Returns:
            Priority Score (0-100)
        """
        weights = self.config['prioritization']['score_weights']
        
        # Churn-Wahrscheinlichkeiten (1 - Survival)
        p6 = 1 - survival_probs.get(6, 1.0)
        p12 = 1 - survival_probs.get(12, 1.0)
        
        # Months-to-Live Komponente (normalisiert, invertiert)
        mtl_component = 1 / (1 + months_to_live / 12)  # Je niedriger MTL, desto höher Score
        
        # Gewichtete Kombination
        score = (
            weights['p6_weight'] * p6 + 
            weights['p12_weight'] * p12 + 
            weights['mtl_weight'] * mtl_component
        )
        
        # Auf 0-100 Skala
        score = score * 100
        
        # Clipping
        score_range = self.config['prioritization']['priority_score_range']
        score = max(score_range[0], min(score_range[1], score))
        
        return float(score)
    
    def _categorize_risk_level(self, priority_score: float) -> str:
        """Kategorisiert Risiko-Level basierend auf Priority Score"""
        if priority_score >= 90:
            return "Very High"
        elif priority_score >= 70:
            return "High"
        elif priority_score >= 50:
            return "Medium"
        elif priority_score >= 20:
            return "Low"
        else:
            return "Very Low"
    
    def _log_prioritization_stats(self, prioritization_df: pd.DataFrame):
        """Protokolliert Priorisierungs-Statistiken"""
        total_customers = len(prioritization_df)
        
        # Risk-Level Verteilung
        risk_distribution = prioritization_df['RiskLevel'].value_counts()
        
        # Priority Score Statistiken
        score_stats = prioritization_df['PriorityScore'].describe()
        
        self.logger.info("📊 Priorisierungs-Statistiken:")
        self.logger.info(f"   👥 Gesamt-Kunden: {total_customers}")
        self.logger.info(f"   📊 Priority Score Range: {score_stats['min']:.1f} - {score_stats['max']:.1f}")
        self.logger.info(f"   📊 Priority Score Median: {score_stats['50%']:.1f}")
        
        self.logger.info("📊 Risiko-Verteilung:")
        for risk_level, count in risk_distribution.items():
            percentage = (count / total_customers) * 100
            self.logger.info(f"   {risk_level}: {count} ({percentage:.1f}%)")
    
    # =============================================================================
    # SURVIVAL ANALYSIS
    # =============================================================================
    
    def generate_survival_curves(self, model: CoxPHFitter,
                               data: pd.DataFrame,
                               stratify_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Generiert Survival-Kurven
        
        Args:
            model: Cox-Modell
            data: Daten für Kurven-Erstellung
            stratify_by: Feature für Stratifizierung (optional)
            
        Returns:
            Survival-Kurven-Daten
        """
        self.logger.info("📈 Generiere Survival-Kurven")
        
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines erforderlich für Survival-Kurven")
        
        survival_data = {
            'baseline_survival': None,
            'stratified_curves': {},
            'median_survival_times': {},
            'survival_statistics': {}
        }
        
        try:
            # Feature-Spalten identifizieren
            feature_cols = [col for col in data.columns 
                           if col not in ['duration', 'event', 'Kunde', 'I_TIMEBASE']]
            
            # Baseline Survival-Funktion
            baseline_sf = model.baseline_survival_
            survival_data['baseline_survival'] = {
                'times': baseline_sf.index.tolist(),
                'probabilities': baseline_sf.values.tolist()
            }
            
            # Median Survival Time
            median_time = None
            for time, prob in zip(baseline_sf.index, baseline_sf.values):
                if prob <= 0.5:
                    median_time = time
                    break
            survival_data['median_survival_times']['baseline'] = median_time
            
            # Stratifizierte Kurven (falls gewünscht)
            if stratify_by and stratify_by in data.columns:
                self.logger.info(f"   📊 Stratifizierung nach {stratify_by}")
                
                unique_values = data[stratify_by].unique()
                for value in unique_values:
                    if pd.notna(value):
                        subset_data = data[data[stratify_by] == value]
                        if len(subset_data) > 5:  # Mindestens 5 Observationen
                            
                            # Durchschnittliche Features für diese Gruppe
                            avg_features = subset_data[feature_cols].mean().to_frame().T
                            
                            # Survival-Funktion für diese Gruppe
                            group_sf = model.predict_survival_function(avg_features)
                            group_sf = group_sf.iloc[:, 0]
                            
                            survival_data['stratified_curves'][str(value)] = {
                                'times': group_sf.index.tolist(),
                                'probabilities': group_sf.values.tolist(),
                                'sample_size': len(subset_data)
                            }
                            
                            # Median für diese Gruppe
                            group_median = None
                            for time, prob in zip(group_sf.index, group_sf.values):
                                if prob <= 0.5:
                                    group_median = time
                                    break
                            survival_data['median_survival_times'][str(value)] = group_median
            
            # Survival-Statistiken
            survival_data['survival_statistics'] = {
                'total_observations': len(data),
                'total_events': int(data['event'].sum()),
                'event_rate': float(data['event'].mean()),
                'max_follow_up': float(data['duration'].max()),
                'median_follow_up': float(data['duration'].median())
            }
            
        except Exception as e:
            self.logger.error(f"❌ Survival-Kurven Generierung fehlgeschlagen: {e}")
            survival_data['error'] = str(e)
        
        # Ergebnisse speichern
        self.survival_curves = survival_data
        
        self.logger.info("✅ Survival-Kurven generiert")
        return survival_data
    
    def analyze_risk_groups(self, prioritization_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analysiert Risiko-Gruppen
        
        Args:
            prioritization_data: Priorisierungs-Daten (optional)
            
        Returns:
            Risiko-Gruppen-Analyse
        """
        if prioritization_data is None:
            prioritization_data = self.prioritization_results
        
        if prioritization_data is None or len(prioritization_data) == 0:
            raise ValueError("Keine Priorisierungs-Daten für Risiko-Analyse verfügbar")
        
        self.logger.info("🎯 Analysiere Risiko-Gruppen")
        
        # Risiko-Gruppen definieren
        def assign_risk_group(score):
            if score >= 90:
                return "Critical"
            elif score >= 70:
                return "High"
            elif score >= 50:
                return "Medium"
            elif score >= 20:
                return "Low"
            else:
                return "Minimal"
        
        # Risiko-Gruppen zuweisen
        analysis_data = prioritization_data.copy()
        analysis_data['RiskGroup'] = analysis_data['PriorityScore'].apply(assign_risk_group)
        
        # Gruppen-Statistiken
        risk_analysis = {
            'risk_groups': {},
            'overall_statistics': {},
            'recommendations': []
        }
        
        total_customers = len(analysis_data)
        
        for risk_group in analysis_data['RiskGroup'].unique():
            group_data = analysis_data[analysis_data['RiskGroup'] == risk_group]
            
            group_stats = {
                'customer_count': len(group_data),
                'percentage': (len(group_data) / total_customers) * 100,
                'avg_priority_score': group_data['PriorityScore'].mean(),
                'avg_p6_risk': group_data['P_Event_6m'].mean(),
                'avg_p12_risk': group_data['P_Event_12m'].mean(),
                'avg_months_to_live': group_data['MonthsToLive_Conditional'].mean(),
                'priority_score_range': {
                    'min': group_data['PriorityScore'].min(),
                    'max': group_data['PriorityScore'].max()
                }
            }
            
            risk_analysis['risk_groups'][risk_group] = group_stats
        
        # Overall-Statistiken
        risk_analysis['overall_statistics'] = {
            'total_customers': total_customers,
            'high_risk_customers': len(analysis_data[analysis_data['PriorityScore'] >= 70]),
            'medium_risk_customers': len(analysis_data[
                (analysis_data['PriorityScore'] >= 50) & (analysis_data['PriorityScore'] < 70)
            ]),
            'low_risk_customers': len(analysis_data[analysis_data['PriorityScore'] < 50]),
            'avg_priority_score': analysis_data['PriorityScore'].mean(),
            'median_priority_score': analysis_data['PriorityScore'].median()
        }
        
        # Business-Empfehlungen
        high_risk_pct = (risk_analysis['overall_statistics']['high_risk_customers'] / total_customers) * 100
        
        if high_risk_pct > 20:
            risk_analysis['recommendations'].append(
                "Hoher Anteil risikanter Kunden (>20%) - Sofortige Retention-Maßnahmen erforderlich"
            )
        
        if high_risk_pct > 10:
            risk_analysis['recommendations'].append(
                "Priorisierte Kontaktaufnahme mit Critical/High-Risk Kunden innerhalb von 7 Tagen"
            )
        
        if risk_analysis['overall_statistics']['avg_priority_score'] > 40:
            risk_analysis['recommendations'].append(
                "Überdurchschnittliches Gesamt-Risiko - Retention-Strategie überprüfen"
            )
        
        risk_analysis['recommendations'].append(
            f"Fokus auf {risk_analysis['overall_statistics']['high_risk_customers']} High-Risk Kunden für maximalen ROI"
        )
        
        self.logger.info(f"✅ Risiko-Analyse abgeschlossen:")
        self.logger.info(f"   🔴 High-Risk: {risk_analysis['overall_statistics']['high_risk_customers']} ({high_risk_pct:.1f}%)")
        self.logger.info(f"   🟡 Medium-Risk: {risk_analysis['overall_statistics']['medium_risk_customers']}")
        self.logger.info(f"   🟢 Low-Risk: {risk_analysis['overall_statistics']['low_risk_customers']}")
        
        return risk_analysis
    
    # =============================================================================
    # VISUALIZATION
    # =============================================================================
    
    def plot_feature_importance(self, model: Optional[CoxPHFitter] = None,
                              top_n: int = 20,
                              output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plottet Feature-Importance
        
        Args:
            model: Cox-Modell (optional)
            top_n: Anzahl Top-Features
            output_path: Ausgabe-Pfad
            
        Returns:
            Pfad zur gespeicherten Grafik (oder None wenn matplotlib nicht verfügbar)
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("⚠️ matplotlib nicht verfügbar - Feature-Importance Plot übersprungen")
            return None
        
        if model is None and self.feature_importance is None:
            self.logger.warning("⚠️ Kein Modell oder Feature-Importance für Plot verfügbar")
            return None
        
        self.logger.info(f"📊 Erstelle Feature-Importance Plot (Top {top_n})")
        
        # Feature-Importance extrahieren falls nötig
        if self.feature_importance is None and model is not None:
            self._extract_feature_importance(model)
        
        if self.feature_importance is None:
            return None
        
        try:
            # Plot erstellen
            plt.style.use(self.config['visualization']['style'])
            fig, ax = plt.subplots(figsize=self.config['visualization']['figsize'])
            
            # Top-N Features
            top_features = self.feature_importance.head(top_n)
            
            # Horizontal Bar Plot
            bars = ax.barh(range(len(top_features)), top_features['abs_coefficient'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Absolute Coefficient')
            ax.set_title(f'Top {top_n} Feature Importance (Cox Model)')
            
            # Farben nach Signifikanz
            colors = ['red' if sig else 'lightblue' for sig in top_features['significant']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Legende
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Significant (p<0.05)')
            blue_patch = mpatches.Patch(color='lightblue', label='Not Significant')
            ax.legend(handles=[red_patch, blue_patch])
            
            plt.tight_layout()
            
            # Speichern
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"feature_importance_{timestamp}.{self.config['visualization']['plot_format']}"
                output_path = self.visualization_dir / filename
            
            plt.savefig(
                output_path,
                dpi=self.config['visualization']['plot_dpi'],
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info(f"📊 Feature-Importance Plot gespeichert: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Feature-Importance Plot fehlgeschlagen: {e}")
            return None
    
    def plot_risk_distribution(self, prioritization_data: Optional[pd.DataFrame] = None,
                             output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plottet Risiko-Verteilung
        
        Args:
            prioritization_data: Priorisierungs-Daten
            output_path: Ausgabe-Pfad
            
        Returns:
            Pfad zur gespeicherten Grafik
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("⚠️ matplotlib nicht verfügbar - Risiko-Verteilung Plot übersprungen")
            return None
        
        if prioritization_data is None:
            prioritization_data = self.prioritization_results
        
        if prioritization_data is None or len(prioritization_data) == 0:
            self.logger.warning("⚠️ Keine Priorisierungs-Daten für Risiko-Plot verfügbar")
            return None
        
        self.logger.info("📊 Erstelle Risiko-Verteilung Plot")
        
        try:
            plt.style.use(self.config['visualization']['style'])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Subplot 1: Priority Score Histogram
            ax1.hist(prioritization_data['PriorityScore'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Priority Score')
            ax1.set_ylabel('Number of Customers')
            ax1.set_title('Distribution of Priority Scores')
            ax1.grid(True, alpha=0.3)
            
            # Statistiken hinzufügen
            mean_score = prioritization_data['PriorityScore'].mean()
            median_score = prioritization_data['PriorityScore'].median()
            ax1.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.1f}')
            ax1.axvline(median_score, color='orange', linestyle='--', label=f'Median: {median_score:.1f}')
            ax1.legend()
            
            # Subplot 2: Risk Level Pie Chart
            risk_counts = prioritization_data['RiskLevel'].value_counts()
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(risk_counts)]
            
            ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            ax2.set_title('Risk Level Distribution')
            
            plt.tight_layout()
            
            # Speichern
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"risk_distribution_{timestamp}.{self.config['visualization']['plot_format']}"
                output_path = self.visualization_dir / filename
            
            plt.savefig(
                output_path,
                dpi=self.config['visualization']['plot_dpi'],
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info(f"📊 Risiko-Verteilung Plot gespeichert: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Risiko-Verteilung Plot fehlgeschlagen: {e}")
            return None
    
    def plot_survival_curves(self, survival_data: Optional[Dict[str, Any]] = None,
                           output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plottet Survival-Kurven
        
        Args:
            survival_data: Survival-Kurven-Daten
            output_path: Ausgabe-Pfad
            
        Returns:
            Pfad zur gespeicherten Grafik
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("⚠️ matplotlib nicht verfügbar - Survival-Kurven Plot übersprungen")
            return None
        
        if survival_data is None:
            survival_data = self.survival_curves
        
        if not survival_data or 'baseline_survival' not in survival_data:
            self.logger.warning("⚠️ Keine Survival-Daten für Plot verfügbar")
            return None
        
        self.logger.info("📈 Erstelle Survival-Kurven Plot")
        
        try:
            plt.style.use(self.config['visualization']['style'])
            fig, ax = plt.subplots(figsize=self.config['visualization']['figsize'])
            
            # Baseline Survival-Kurve
            baseline = survival_data['baseline_survival']
            ax.plot(baseline['times'], baseline['probabilities'], 
                   linewidth=2, label='Baseline Survival', color='blue')
            
            # Stratifizierte Kurven (falls vorhanden)
            if 'stratified_curves' in survival_data and survival_data['stratified_curves']:
                colors = plt.cm.Set1(np.linspace(0, 1, len(survival_data['stratified_curves'])))
                
                for i, (group, curve_data) in enumerate(survival_data['stratified_curves'].items()):
                    ax.plot(curve_data['times'], curve_data['probabilities'], 
                           linewidth=2, label=f'Group {group} (n={curve_data["sample_size"]})',
                           color=colors[i])
            
            # Plot-Formatierung
            ax.set_xlabel('Time (Months)')
            ax.set_ylabel('Survival Probability')
            ax.set_title('Cox Model Survival Curves')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Median-Linien hinzufügen
            if 'median_survival_times' in survival_data:
                for group, median_time in survival_data['median_survival_times'].items():
                    if median_time is not None:
                        ax.axvline(median_time, linestyle='--', alpha=0.7,
                                 label=f'Median {group}: {median_time:.1f}m')
            
            plt.tight_layout()
            
            # Speichern
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"survival_curves_{timestamp}.{self.config['visualization']['plot_format']}"
                output_path = self.visualization_dir / filename
            
            plt.savefig(
                output_path,
                dpi=self.config['visualization']['plot_dpi'],
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info(f"📈 Survival-Kurven Plot gespeichert: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Survival-Kurven Plot fehlgeschlagen: {e}")
            return None
    
    # =============================================================================
    # BUSINESS REPORTING
    # =============================================================================
    
    def generate_business_report(self, prioritization_data: Optional[pd.DataFrame] = None,
                               model_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generiert Business-Report
        
        Args:
            prioritization_data: Priorisierungs-Daten
            model_performance: Model-Performance-Daten
            
        Returns:
            Business-Report
        """
        self.logger.info("💼 Generiere Business-Report")
        
        if prioritization_data is None:
            prioritization_data = self.prioritization_results
        
        if model_performance is None:
            model_performance = self.evaluation_results.get('performance', {})
        
        # Executive Summary
        executive_summary = self._create_executive_summary(prioritization_data, model_performance)
        
        # KPI-Metriken
        kpi_metrics = self._calculate_business_kpis(prioritization_data)
        
        # Actionable Insights
        actionable_insights = self._generate_actionable_insights(prioritization_data)
        
        # Risk Alerts
        risk_alerts = self._generate_risk_alerts(prioritization_data)
        
        # Recommendations
        recommendations = self._generate_business_recommendations(prioritization_data, model_performance)
        
        business_report = {
            'metadata': {
                'report_type': 'Cox_Business_Report',
                'generated_at': datetime.now().isoformat(),
                'report_period': f"Analysis as of cutoff",
                'total_customers_analyzed': len(prioritization_data) if prioritization_data is not None else 0
            },
            'executive_summary': executive_summary,
            'kpi_metrics': kpi_metrics,
            'actionable_insights': actionable_insights,
            'risk_alerts': risk_alerts,
            'recommendations': recommendations,
            'model_performance_summary': self._summarize_model_performance(model_performance)
        }
        
        self.logger.info("✅ Business-Report generiert")
        return business_report
    
    def _create_executive_summary(self, prioritization_data: Optional[pd.DataFrame], 
                                model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Executive Summary"""
        if prioritization_data is None or len(prioritization_data) == 0:
            return {
                'status': 'No prioritization data available',
                'key_findings': ['Insufficient data for analysis'],
                'urgent_actions': ['Generate customer prioritization first']
            }
        
        total_customers = len(prioritization_data)
        high_risk_count = len(prioritization_data[prioritization_data['PriorityScore'] >= 70])
        critical_risk_count = len(prioritization_data[prioritization_data['PriorityScore'] >= 90])
        
        avg_priority = prioritization_data['PriorityScore'].mean()
        avg_6m_risk = prioritization_data['P_Event_6m'].mean()
        avg_12m_risk = prioritization_data['P_Event_12m'].mean()
        
        # Model Performance
        c_index = model_performance.get('basic_metrics', {}).get('concordance_index', 0.5)
        
        summary = {
            'total_customers': total_customers,
            'model_performance': f"{c_index:.3f} C-Index",
            'overall_risk_level': 'High' if avg_priority > 50 else 'Medium' if avg_priority > 25 else 'Low',
            'key_findings': [
                f"{critical_risk_count} customers in Critical risk category (≥90 Priority Score)",
                f"{high_risk_count} customers in High risk category (≥70 Priority Score)",
                f"Average 6-month churn probability: {avg_6m_risk:.1%}",
                f"Average 12-month churn probability: {avg_12m_risk:.1%}",
                f"Model performance: {c_index:.3f} (Excellent)" if c_index >= 0.9 else f"Model performance: {c_index:.3f} (Good)" if c_index >= 0.8 else f"Model performance: {c_index:.3f} (Fair)"
            ],
            'urgent_actions': []
        }
        
        # Urgent Actions basierend auf Risiko-Level
        if critical_risk_count > 0:
            summary['urgent_actions'].append(f"Immediate intervention required for {critical_risk_count} critical risk customers")
        
        if high_risk_count > total_customers * 0.15:  # >15% high risk
            summary['urgent_actions'].append("High proportion of at-risk customers - review retention strategy")
        
        if avg_6m_risk > 0.2:  # >20% average 6-month risk
            summary['urgent_actions'].append("Elevated short-term churn risk - accelerate retention efforts")
        
        return summary
    
    def _calculate_business_kpis(self, prioritization_data: pd.DataFrame) -> Dict[str, Any]:
        """Berechnet Business-KPIs"""
        if prioritization_data is None or len(prioritization_data) == 0:
            return {}
        
        total_customers = len(prioritization_data)
        
        kpis = {
            'customer_metrics': {
                'total_customers': total_customers,
                'critical_risk_customers': len(prioritization_data[prioritization_data['PriorityScore'] >= 90]),
                'high_risk_customers': len(prioritization_data[prioritization_data['PriorityScore'] >= 70]),
                'medium_risk_customers': len(prioritization_data[
                    (prioritization_data['PriorityScore'] >= 50) & (prioritization_data['PriorityScore'] < 70)
                ]),
                'low_risk_customers': len(prioritization_data[prioritization_data['PriorityScore'] < 50])
            },
            'risk_metrics': {
                'avg_priority_score': float(prioritization_data['PriorityScore'].mean()),
                'median_priority_score': float(prioritization_data['PriorityScore'].median()),
                'max_priority_score': float(prioritization_data['PriorityScore'].max()),
                'avg_6m_churn_prob': float(prioritization_data['P_Event_6m'].mean()),
                'avg_12m_churn_prob': float(prioritization_data['P_Event_12m'].mean()),
                'avg_months_to_live': float(prioritization_data['MonthsToLive_Conditional'].mean())
            },
            'percentile_analysis': {
                'top_1_percent_threshold': float(prioritization_data['PriorityScore'].quantile(0.99)),
                'top_5_percent_threshold': float(prioritization_data['PriorityScore'].quantile(0.95)),
                'top_10_percent_threshold': float(prioritization_data['PriorityScore'].quantile(0.90)),
                'top_1_percent_count': int(total_customers * 0.01),
                'top_5_percent_count': int(total_customers * 0.05),
                'top_10_percent_count': int(total_customers * 0.10)
            }
        }
        
        return kpis
    
    def _generate_actionable_insights(self, prioritization_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generiert actionable Insights"""
        if prioritization_data is None or len(prioritization_data) == 0:
            return []
        
        insights = []
        
        # Insight 1: Immediate attention customers
        critical_customers = prioritization_data[prioritization_data['PriorityScore'] >= 90]
        if len(critical_customers) > 0:
            insights.append({
                'type': 'immediate_action',
                'priority': 'critical',
                'title': 'Critical Risk Customers Require Immediate Attention',
                'description': f"{len(critical_customers)} customers have Priority Scores ≥90",
                'action': f"Contact these {len(critical_customers)} customers within 48 hours",
                'potential_impact': 'Prevent immediate churn of highest-risk customers'
            })
        
        # Insight 2: High-value intervention targets
        high_risk_customers = prioritization_data[
            (prioritization_data['PriorityScore'] >= 70) & (prioritization_data['PriorityScore'] < 90)
        ]
        if len(high_risk_customers) > 0:
            insights.append({
                'type': 'strategic_action',
                'priority': 'high',
                'title': 'High-Risk Customer Intervention Program',
                'description': f"{len(high_risk_customers)} customers in high-risk category (70-89 Priority Score)",
                'action': f"Develop targeted retention campaigns for {len(high_risk_customers)} customers",
                'potential_impact': 'Systematic reduction of churn risk through proactive engagement'
            })
        
        # Insight 3: Short-term risk analysis
        high_6m_risk = prioritization_data[prioritization_data['P_Event_6m'] > 0.3]
        if len(high_6m_risk) > 0:
            insights.append({
                'type': 'timeline_action',
                'priority': 'high',
                'title': 'Short-Term Churn Risk Alert',
                'description': f"{len(high_6m_risk)} customers have >30% probability of churning within 6 months",
                'action': 'Accelerate retention efforts for immediate-term at-risk customers',
                'potential_impact': 'Reduce short-term churn through timely intervention'
            })
        
        # Insight 4: Resource allocation
        total_high_risk = len(prioritization_data[prioritization_data['PriorityScore'] >= 50])
        total_customers = len(prioritization_data)
        if total_high_risk > 0:
            insights.append({
                'type': 'resource_allocation',
                'priority': 'medium',
                'title': 'Retention Resource Allocation',
                'description': f"{total_high_risk} customers ({total_high_risk/total_customers:.1%}) require retention focus",
                'action': f"Allocate retention resources to top {total_high_risk} risk customers",
                'potential_impact': 'Optimize retention ROI through focused resource allocation'
            })
        
        return insights
    
    def _generate_risk_alerts(self, prioritization_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generiert Risiko-Alerts"""
        if prioritization_data is None or len(prioritization_data) == 0:
            return []
        
        alerts = []
        total_customers = len(prioritization_data)
        
        # Alert 1: High proportion of critical risk customers
        critical_count = len(prioritization_data[prioritization_data['PriorityScore'] >= 90])
        if critical_count / total_customers > 0.05:  # >5% critical
            alerts.append({
                'level': 'critical',
                'type': 'high_risk_proportion',
                'message': f"High proportion of critical risk customers: {critical_count}/{total_customers} ({critical_count/total_customers:.1%})",
                'threshold': '5%',
                'action_required': 'Immediate management attention and resource allocation'
            })
        
        # Alert 2: Elevated average risk
        avg_priority = prioritization_data['PriorityScore'].mean()
        if avg_priority > 40:
            alerts.append({
                'level': 'warning',
                'type': 'elevated_average_risk',
                'message': f"Elevated average Priority Score: {avg_priority:.1f}",
                'threshold': '40',
                'action_required': 'Review and strengthen overall retention strategy'
            })
        
        # Alert 3: High short-term churn probability
        avg_6m_risk = prioritization_data['P_Event_6m'].mean()
        if avg_6m_risk > 0.15:  # >15% average 6-month churn risk
            alerts.append({
                'level': 'warning',
                'type': 'high_short_term_risk',
                'message': f"High average 6-month churn probability: {avg_6m_risk:.1%}",
                'threshold': '15%',
                'action_required': 'Accelerate short-term retention initiatives'
            })
        
        # Alert 4: Low months-to-live
        avg_mtl = prioritization_data['MonthsToLive_Conditional'].mean()
        if avg_mtl < 18:  # <18 months average
            alerts.append({
                'level': 'info',
                'type': 'reduced_customer_lifetime',
                'message': f"Reduced average customer lifetime: {avg_mtl:.1f} months",
                'threshold': '18 months',
                'action_required': 'Focus on long-term value preservation strategies'
            })
        
        return alerts
    
    def _generate_business_recommendations(self, prioritization_data: pd.DataFrame, 
                                         model_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generiert Business-Empfehlungen"""
        recommendations = []
        
        if prioritization_data is None or len(prioritization_data) == 0:
            recommendations.append({
                'category': 'data_quality',
                'priority': 'high',
                'title': 'Generate Customer Prioritization',
                'description': 'No prioritization data available for business recommendations',
                'implementation': 'Run customer prioritization analysis first'
            })
            return recommendations
        
        # Model Performance Recommendations
        c_index = model_performance.get('basic_metrics', {}).get('concordance_index', 0.5)
        if c_index < 0.8:
            recommendations.append({
                'category': 'model_improvement',
                'priority': 'medium',
                'title': 'Improve Model Performance',
                'description': f"Current C-Index {c_index:.3f} below optimal threshold (0.8+)",
                'implementation': 'Enhance feature engineering or collect additional predictive data'
            })
        
        # Customer Segmentation Recommendations
        total_customers = len(prioritization_data)
        high_risk_count = len(prioritization_data[prioritization_data['PriorityScore'] >= 70])
        
        if high_risk_count > total_customers * 0.2:  # >20% high risk
            recommendations.append({
                'category': 'retention_strategy',
                'priority': 'high',
                'title': 'Implement Tiered Retention Strategy',
                'description': f"High proportion ({high_risk_count/total_customers:.1%}) of at-risk customers",
                'implementation': 'Develop differentiated retention approaches by risk level'
            })
        
        # Resource Allocation Recommendations
        critical_count = len(prioritization_data[prioritization_data['PriorityScore'] >= 90])
        if critical_count > 0:
            recommendations.append({
                'category': 'resource_allocation',
                'priority': 'critical',
                'title': 'Emergency Retention Protocol',
                'description': f"{critical_count} customers in critical risk category",
                'implementation': f"Assign dedicated account managers to {critical_count} critical customers"
            })
        
        # Monitoring Recommendations
        recommendations.append({
            'category': 'monitoring',
            'priority': 'medium',
            'title': 'Implement Continuous Risk Monitoring',
            'description': 'Regular model updates and performance tracking',
            'implementation': 'Schedule monthly model retraining and quarterly performance reviews'
        })
        
        # Business Process Recommendations
        avg_6m_risk = prioritization_data['P_Event_6m'].mean()
        if avg_6m_risk > 0.1:
            recommendations.append({
                'category': 'business_process',
                'priority': 'medium',
                'title': 'Proactive Customer Success Program',
                'description': f"Average 6-month churn risk of {avg_6m_risk:.1%} indicates need for proactive engagement",
                'implementation': 'Implement early warning system and proactive outreach protocols'
            })
        
        return recommendations
    
    def _summarize_model_performance(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Zusammenfassung der Model-Performance für Business-Report"""
        basic_metrics = model_performance.get('basic_metrics', {})
        
        summary = {
            'c_index': basic_metrics.get('concordance_index', 0.5),
            'performance_level': 'Unknown',
            'model_quality': model_performance.get('model_diagnostics', {}).get('overall_quality', 'unknown'),
            'total_features': basic_metrics.get('number_of_features', 0),
            'sample_size': basic_metrics.get('number_of_observations', 0),
            'event_rate': basic_metrics.get('event_rate', 0)
        }
        
        # Performance Level
        c_index = summary['c_index']
        if c_index >= 0.9:
            summary['performance_level'] = 'Excellent'
        elif c_index >= 0.8:
            summary['performance_level'] = 'Good'
        elif c_index >= 0.7:
            summary['performance_level'] = 'Fair'
        else:
            summary['performance_level'] = 'Poor'
        
        return summary
    
    # =============================================================================
    # EXPORT & PERSISTENCE
    # =============================================================================
    
    def save_prioritization_results(self, data: Optional[pd.DataFrame] = None,
                                  output_path: Optional[Path] = None,
                                  format: str = 'csv') -> Path:
        """
        Speichert Priorisierungs-Ergebnisse
        
        Args:
            data: Priorisierungs-DataFrame (optional)
            output_path: Ausgabe-Pfad (optional)
            format: Format ('csv', 'json', 'excel')
            
        Returns:
            Pfad zur gespeicherten Datei
        """
        if data is None:
            data = self.prioritization_results
        
        if data is None or len(data) == 0:
            raise ValueError("Keine Priorisierungs-Daten zum Speichern verfügbar")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"prioritization_{timestamp}.{format}"
            output_path = self.prioritization_dir / filename
        
        self.logger.info(f"💾 Speichere Priorisierungsdaten: {output_path}")
        
        try:
            if format == 'csv':
                data.to_csv(output_path, sep=';', index=False, encoding='utf-8')
            elif format == 'json':
                data.to_json(output_path, orient='records', indent=2)
            elif format == 'excel':
                data.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unbekanntes Format: {format}")
            
            self.logger.info(f"✅ Priorisierungsdaten gespeichert: {len(data)} Records")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Speichern fehlgeschlagen: {e}")
            raise
    
    def export_evaluation_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Exportiert vollständigen Evaluation-Report
        
        Args:
            output_path: Ausgabe-Pfad (optional)
            
        Returns:
            Pfad zum Evaluation-Report (JSON)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cox_evaluation_report_{timestamp}.json"
            output_path = self.output_dir / filename
        
        # Vollständiger Report
        evaluation_report = {
            'metadata': {
                'report_type': 'Cox_Evaluation_Report',
                'generated_at': datetime.now().isoformat(),
                'evaluator_config': self.config
            },
            'evaluation_results': self.evaluation_results,
            'prioritization_summary': self._create_prioritization_summary(),
            'survival_curves': self.survival_curves,
            'feature_importance_summary': self._create_feature_importance_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📄 Evaluation-Report exportiert: {output_path}")
        return output_path
    
    def _create_prioritization_summary(self) -> Dict[str, Any]:
        """Erstellt Priorisierungs-Summary"""
        if self.prioritization_results is None:
            return {'status': 'no_data'}
        
        data = self.prioritization_results
        
        return {
            'total_customers': len(data),
            'risk_distribution': data['RiskLevel'].value_counts().to_dict(),
            'priority_score_stats': {
                'mean': float(data['PriorityScore'].mean()),
                'median': float(data['PriorityScore'].median()),
                'min': float(data['PriorityScore'].min()),
                'max': float(data['PriorityScore'].max()),
                'std': float(data['PriorityScore'].std())
            },
            'churn_probability_stats': {
                '6_month_avg': float(data['P_Event_6m'].mean()),
                '12_month_avg': float(data['P_Event_12m'].mean())
            }
        }
    
    def _create_feature_importance_summary(self) -> Dict[str, Any]:
        """Erstellt Feature-Importance-Summary"""
        if self.feature_importance is None:
            return {'status': 'no_data'}
        
        top_5 = self.feature_importance.head(5)
        
        return {
            'total_features': len(self.feature_importance),
            'significant_features': int(self.feature_importance['significant'].sum()),
            'top_5_features': [
                {
                    'feature': row['feature'],
                    'coefficient': float(row['coefficient']),
                    'hazard_ratio': float(row['hazard_ratio']),
                    'p_value': float(row['p_value']),
                    'significant': bool(row['significant'])
                }
                for _, row in top_5.iterrows()
            ]
        }


if __name__ == "__main__":
    # Test des Cox Evaluators
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Cox Evaluator Test")
    
    # Evaluator initialisieren
    evaluator = CoxEvaluator()
    
    # Test-Daten erstellen
    np.random.seed(42)
    n_customers = 50
    
    test_prioritization = pd.DataFrame({
        'Kunde': range(1, n_customers + 1),
        'P_Event_6m': np.random.beta(2, 8, n_customers),  # Niedrige Churn-Wahrscheinlichkeiten
        'P_Event_12m': np.random.beta(3, 7, n_customers),
        'RMST_12m': np.random.normal(10, 2, n_customers),
        'RMST_24m': np.random.normal(20, 4, n_customers),
        'MonthsToLive_Conditional': np.random.exponential(24, n_customers),
        'PriorityScore': np.random.gamma(2, 20, n_customers),  # 0-100 Range
        'RiskLevel': np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_customers)
    })
    
    # Clipping für realistische Werte
    test_prioritization['PriorityScore'] = np.clip(test_prioritization['PriorityScore'], 0, 100)
    
    evaluator.prioritization_results = test_prioritization
    
    try:
        # Test: Risiko-Analyse
        risk_analysis = evaluator.analyze_risk_groups()
        print(f"✅ Risiko-Analyse erfolgreich: {len(risk_analysis['risk_groups'])} Gruppen")
        
        # Test: Business-Report
        business_report = evaluator.generate_business_report()
        print(f"💼 Business-Report generiert: {len(business_report['actionable_insights'])} Insights")
        
        # Test: Priorisierung speichern
        output_path = evaluator.save_prioritization_results(format='csv')
        print(f"💾 Priorisierung gespeichert: {output_path}")
        
        print("✅ Alle Tests erfolgreich!")
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
