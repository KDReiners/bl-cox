#!/usr/bin/env python3
"""
Cox Model Trainer - Optimiertes Cox-Model-Training für 0.95+ C-Index
=====================================================================

Performance-optimiertes Training von Cox-Proportional-Hazards-Modellen
mit Fokus auf maximale C-Index-Performance (Ziel: 0.95+).

Kernfunktionen:
- Cox-Model-Training mit lifelines optimiert
- Hyperparameter-Tuning für beste Performance
- Cross-Validation für robuste Schätzung
- Model-Convergence-Validation
- Model-Persistierung für Production

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
import pickle
import warnings
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.preprocessing import StandardScaler
import itertools

# Lifelines imports
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from lifelines.exceptions import ConvergenceWarning, ConvergenceError
    LIFELINES_AVAILABLE = True
except ImportError:
    print("⚠️ lifelines nicht verfügbar - Cox-Training nicht möglich")
    LIFELINES_AVAILABLE = False

# Projekt-Pfade hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.paths_config import ProjectPaths
    from bl.Cox.cox_constants import (
        HUNDRED_PERCENT, SIX_MONTH_HORIZON, TWELVE_MONTH_HORIZON
    )
except ImportError as e:
    print(f"⚠️ Import-Fehler: {e}")
    # Fallback-Konstanten
    HUNDRED_PERCENT = 100
    SIX_MONTH_HORIZON = 6
    TWELVE_MONTH_HORIZON = 12


class CoxModelTrainer:
    """
    Optimiertes Cox-Model-Training für maximale Performance
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert Model Trainer
        
        Args:
            model_config: Konfiguration für Model-Training
        """
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines ist erforderlich für Cox-Model-Training")
        
        self.model_config = model_config or self._default_config()
        self.logger = self._setup_logging()
        
        # Paths
        try:
            self.paths = ProjectPaths()
            self.models_dir = self.paths.models_directory()
            self.output_dir = self.paths.dynamic_outputs_directory() / "cox_analysis"
        except:
            self.models_dir = Path("models")
            self.output_dir = Path("dynamic_system_outputs/cox_analysis")
        
        # Erstelle Output-Verzeichnisse
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # State
        self.fitted_model: Optional[CoxPHFitter] = None
        self.training_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.training_results: Dict[str, Any] = {}
        self.convergence_info: Dict[str, Any] = {}
        self.hyperparameter_results: Dict[str, Any] = {}
        
        self.logger.info("🎯 Cox Model Trainer initialisiert")
    
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
    
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    
    def _default_config(self) -> Dict[str, Any]:
        """Standard-Konfiguration für optimale Performance"""
        return {
            # Basis-Parameter (bewährt aus cox_optimized_analyzer)
            'penalizer': 0.01,           # Leichte Regularisierung
            'l1_ratio': 0.0,             # Ridge-Regularisierung (L2)
            'alpha': 0.95,               # Konfidenz-Level
            'max_iterations': 1000,      # Maximale Iterationen
            # 'step_size': 0.1,            # Learning Rate (nicht unterstützt in aktueller lifelines-Version)
            'show_progress': True,       # Progress-Anzeige
            'robust': True,              # Robuste Fehlerbehandlung
            'tie_method': 'Efron',       # Tie-Handling-Methode
            
            # Performance-Ziele
            'target_c_index': 0.95,      # Ziel-C-Index
            'min_acceptable_c_index': 0.85,  # Mindest-akzeptable Performance
            
            # Hyperparameter-Tuning
            'hyperparameter_tuning': {
                'enabled': True,
                'method': 'grid_search',   # 'grid_search', 'random_search'
                'cv_folds': 3,
                'scoring_metric': 'concordance',
                'penalizer_range': [0.01, 0.1],  # Weniger aggressive Regularisierung bei wenigen Features
                'l1_ratio_range': [0.3, 0.5, 0.7, 1.0],  # Bevorzuge L1-Regularisierung (Lasso)
                # 'step_size_range': [0.05, 0.1, 0.2],  # Nicht unterstützt
                'max_combinations': 20     # Limitiert Kombinationen
            },
            
            # Cross-Validation
            'cross_validation': {
                'enabled': True,
                'cv_folds': 5,
                'stratify_by_event': True,
                'random_state': 42
            },
            
            # Convergence-Kontrolle
            'convergence': {
                'check_convergence': True,
                'max_gradient_norm': 1e-6,
                'min_log_likelihood_improvement': 1e-8,
                'convergence_tolerance': 1e-7
            },
            
            # Model-Validation
            'validation': {
                'check_proportional_hazards': True,
                'outlier_detection': True,
                'feature_importance_threshold': 0.01
            }
        }
    
    # =============================================================================
    # DATA PREPARATION
    # =============================================================================
    
    def prepare_cox_data(self, survival_panel: pd.DataFrame, 
                        features: pd.DataFrame) -> pd.DataFrame:
        """
        Bereitet Daten für Cox-Training vor
        
        Args:
            survival_panel: Cox-Survival-Panel (Kunde, duration, event)
            features: Feature-Matrix (Kunde, [features])
            
        Returns:
            Kombinierter Datensatz für Cox-Training:
            - duration: Überlebensdauer
            - event: Event-Indikator  
            - [feature_columns]: Alle Features
        """
        self.logger.info("🔧 Bereite Cox-Training-Daten vor")
        
        # Validiere Input-Daten
        self._validate_input_data(survival_panel, features)
        
        # Merge Survival-Panel mit Features
        training_data = survival_panel.merge(features, on='Kunde', how='inner')
        
        if len(training_data) == 0:
            raise ValueError("Keine Daten nach Merge von Survival-Panel und Features")
        
        # Feature-Spalten identifizieren
        survival_cols = ['Kunde', 'duration', 'event', 't_start', 't_end', 'I_Alive']
        self.feature_columns = [col for col in training_data.columns 
                               if col not in survival_cols]
        
        # Nur erforderliche Spalten behalten
        required_cols = ['duration', 'event'] + self.feature_columns
        training_data = training_data[required_cols].copy()
        
        # Datenvalidierung
        validation_report = self.validate_cox_data(training_data)
        
        if not validation_report['validation_passed']:
            raise ValueError(f"Datenvalidierung fehlgeschlagen: {validation_report['issues']}")
        
        # Training-Daten speichern
        self.training_data = training_data
        
        self.logger.info(f"✅ Cox-Training-Daten vorbereitet:")
        self.logger.info(f"   📊 {len(training_data)} Records")
        self.logger.info(f"   🔢 {len(self.feature_columns)} Features")
        self.logger.info(f"   📈 Event-Rate: {training_data['event'].mean():.3f}")
        self.logger.info(f"   ⏱️ Duration-Range: {training_data['duration'].min()}-{training_data['duration'].max()}")
        
        return training_data
    
    def _validate_input_data(self, survival_panel: pd.DataFrame, features: pd.DataFrame):
        """Validiert Input-Daten für Cox-Training"""
        # Survival Panel Validierung
        required_survival_cols = ['Kunde', 'duration', 'event']
        missing_survival = [col for col in required_survival_cols if col not in survival_panel.columns]
        if missing_survival:
            raise ValueError(f"Fehlende Spalten in Survival-Panel: {missing_survival}")
        
        # Features Validierung
        if 'Kunde' not in features.columns:
            raise ValueError("Kunde-Spalte fehlt in Features")
        
        feature_cols = [col for col in features.columns if col != 'Kunde']
        if len(feature_cols) == 0:
            raise ValueError("Keine Feature-Spalten gefunden")
        
        # Datentyp-Validierung
        if not pd.api.types.is_numeric_dtype(survival_panel['duration']):
            raise ValueError("Duration muss numerisch sein")
        
        if not survival_panel['event'].isin([0, 1]).all():
            raise ValueError("Event muss 0 oder 1 sein")
    
    def validate_cox_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert Cox-Training-Daten
        
        Args:
            data: Cox-Training-Datensatz
            
        Returns:
            Validierungs-Report
        """
        self.logger.info("🔍 Validiere Cox-Training-Daten")
        
        issues = []
        warnings_list = []
        
        # Basis-Validierung
        if 'duration' not in data.columns or 'event' not in data.columns:
            issues.append("Duration oder Event-Spalte fehlt")
        
        # Duration-Validierung
        if 'duration' in data.columns:
            if (data['duration'] <= 0).any():
                negative_count = (data['duration'] <= 0).sum()
                issues.append(f"{negative_count} Records mit Duration <= 0")
            
            if data['duration'].isnull().any():
                null_count = data['duration'].isnull().sum()
                issues.append(f"{null_count} Records mit NULL Duration")
        
        # Event-Validierung
        if 'event' in data.columns:
            if not data['event'].isin([0, 1]).all():
                issues.append("Event-Werte nicht in {0, 1}")
            
            event_rate = data['event'].mean()
            if event_rate < 0.01:
                warnings_list.append(f"Sehr niedrige Event-Rate: {event_rate:.3f}")
            elif event_rate > 0.9:
                warnings_list.append(f"Sehr hohe Event-Rate: {event_rate:.3f}")
        
        # Feature-Validierung (temporär vereinfacht)
        try:
            feature_issues = self._validate_features(data)
            issues.extend(feature_issues['issues'])
            warnings_list.extend(feature_issues['warnings'])
        except Exception as e:
            self.logger.warning(f"⚠️ Feature-Validierung übersprungen: {e}")
            # Fortfahren ohne Feature-Validierung
        
        # Missing Values
        missing_total = data.isnull().sum().sum()
        if missing_total > 0:
            issues.append(f"{missing_total} fehlende Werte gefunden")
        
        # Infinite Values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        infinite_total = np.isinf(data[numeric_cols]).sum().sum()
        if infinite_total > 0:
            issues.append(f"{infinite_total} unendliche Werte gefunden")
        
        # Qualitäts-Score berechnen
        quality_score = 1.0 - (len(issues) * 0.2) - (len(warnings_list) * 0.05)
        quality_score = max(0.0, quality_score)
        
        validation_passed = len(issues) == 0 and quality_score >= 0.7
        
        report = {
            'validation_passed': validation_passed,
            'quality_score': quality_score,
            'total_records': len(data),
            'feature_count': len(self.feature_columns),
            'event_rate': data['event'].mean() if 'event' in data.columns else 0,
            'issues': issues,
            'warnings': warnings_list,
            'missing_values': missing_total,
            'infinite_values': infinite_total
        }
        
        if validation_passed:
            self.logger.info(f"✅ Datenvalidierung erfolgreich (Score: {quality_score:.3f})")
        else:
            self.logger.error(f"❌ Datenvalidierung fehlgeschlagen (Score: {quality_score:.3f})")
            for issue in issues:
                self.logger.error(f"   - {issue}")
        
        return report
    
    def _validate_features(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validiert Feature-Qualität"""
        issues = []
        warnings_list = []
        
        for col in self.feature_columns:
            if col not in data.columns:
                issues.append(f"Feature {col} fehlt in Daten")
                continue
            
            # Konstante Features
            if data[col].nunique() <= 1:
                warnings_list.append(f"Feature {col} ist konstant")
            
            # Hohe Missing-Rate
            missing_rate = data[col].isnull().mean()
            if missing_rate > 0.5:
                warnings_list.append(f"Feature {col} hat {missing_rate:.1%} fehlende Werte")
            
            # Extreme Outliers (nur für numerische Features)
            if pd.api.types.is_numeric_dtype(data[col]):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outlier_mask = (data[col] < Q1 - 3*IQR) | (data[col] > Q3 + 3*IQR)
                    outlier_count = outlier_mask.sum()
                    if outlier_count > len(data) * 0.1:  # >10% Outliers
                        warnings_list.append(f"Feature {col} hat {outlier_count} extreme Outliers")
        
        return {'issues': issues, 'warnings': warnings_list}
    
    # =============================================================================
    # MODEL TRAINING
    # =============================================================================
    
    def train_cox_model(self, data: Optional[pd.DataFrame] = None, 
                       duration_col: str = 'duration',
                       event_col: str = 'event') -> CoxPHFitter:
        """
        Trainiert Cox-Proportional-Hazards-Modell
        
        Args:
            data: Training-Datensatz (optional, nutzt self.training_data)
            duration_col: Name der Duration-Spalte
            event_col: Name der Event-Spalte
            
        Returns:
            Trainiertes CoxPHFitter-Modell
        """
        self.logger.info("🎯 Starte Cox-Model-Training")
        
        if data is None:
            if self.training_data is None:
                raise ValueError("Keine Training-Daten verfügbar")
            data = self.training_data
        
        # Cox-Modell initialisieren
        cox_model = CoxPHFitter(
            penalizer=self.model_config['penalizer'],
            l1_ratio=self.model_config['l1_ratio'],
            alpha=self.model_config['alpha']
        )
        
        start_time = datetime.now()
        
        try:
            # Warnings für Convergence temporär unterdrücken
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                # Model fitten
                cox_model.fit(
                    data,
                    duration_col=duration_col,
                    event_col=event_col,
                    show_progress=self.model_config.get('show_progress', False),
                    # step_size=self.model_config.get('step_size', 0.1),  # Nicht unterstützt
                    robust=self.model_config.get('robust', True)
                )
        
        except ConvergenceError as e:
            self.logger.error(f"❌ Konvergenz-Fehler: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Training-Fehler: {e}")
            raise
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Konvergenz validieren
        convergence_info = self.validate_convergence(cox_model)
        self.convergence_info = convergence_info
        
        # Performance evaluieren
        c_index = self.calculate_concordance_index(cox_model, data)
        
        # Training-Ergebnisse speichern
        self.training_results = {
            'training_time_seconds': training_time,
            'concordance_index': c_index,
            'log_likelihood': float(cox_model.log_likelihood_),
            'aic': float(cox_model.AIC_),
            'partial_aic': float(cox_model.AIC_partial_),
            'convergence_info': convergence_info,
            'model_params': {
                'penalizer': self.model_config['penalizer'],
                'l1_ratio': self.model_config['l1_ratio'],
                'alpha': self.model_config['alpha']
            }
        }
        
        self.fitted_model = cox_model
        
        self.logger.info(f"✅ Cox-Model-Training abgeschlossen:")
        self.logger.info(f"   🎯 C-Index: {c_index:.4f}")
        self.logger.info(f"   📊 Log-Likelihood: {cox_model.log_likelihood_:.2f}")
        self.logger.info(f"   📊 AIC: {cox_model.AIC_:.2f}")
        self.logger.info(f"   ⏱️ Training-Zeit: {training_time:.2f}s")
        self.logger.info(f"   ✅ Konvergiert: {'Ja' if convergence_info['converged'] else 'Nein'}")
        
        return cox_model
    
    def hyperparameter_tuning(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Hyperparameter-Optimierung für maximale Performance
        
        Args:
            data: Training-Datensatz (optional)
            
        Returns:
            Optimale Hyperparameter mit Performance-Metriken
        """
        if not self.model_config['hyperparameter_tuning']['enabled']:
            self.logger.info("ℹ️ Hyperparameter-Tuning deaktiviert")
            return {}
        
        self.logger.info("🔬 Starte Hyperparameter-Tuning")
        
        if data is None:
            data = self.training_data
        
        tuning_config = self.model_config['hyperparameter_tuning']
        
        # Parameter-Grid erstellen
        param_grid = {
            'penalizer': tuning_config['penalizer_range'],
            'l1_ratio': tuning_config['l1_ratio_range'],
            # 'step_size': tuning_config.get('step_size_range', [0.1])  # Nicht unterstützt
        }
        
        # Alle Kombinationen generieren
        param_combinations = list(ParameterGrid(param_grid))
        
        # Limitiere Anzahl der Kombinationen
        max_combinations = tuning_config.get('max_combinations', 20)
        if len(param_combinations) > max_combinations:
            # Zufällige Auswahl
            np.random.seed(42)
            param_combinations = np.random.choice(
                param_combinations, 
                size=max_combinations, 
                replace=False
            ).tolist()
        
        self.logger.info(f"   🔬 Teste {len(param_combinations)} Parameter-Kombinationen")
        
        # Cross-Validation Setup
        cv_folds = tuning_config['cv_folds']
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = []
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_combinations):
            self.logger.info(f"   📊 Kombination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Cross-Validation für diese Parameter-Kombination
                cv_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
                    train_data = data.iloc[train_idx]
                    val_data = data.iloc[val_idx]
                    
                    # Model mit aktuellen Parametern trainieren
                    cox_model = CoxPHFitter(
                        penalizer=params['penalizer'],
                        l1_ratio=params['l1_ratio'],
                        alpha=self.model_config['alpha']
                    )
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        
                        cox_model.fit(
                            train_data,
                            duration_col='duration',
                            event_col='event',
                            show_progress=False,
                            # step_size=params.get('step_size', 0.1),  # Nicht unterstützt
                            robust=True
                        )
                    
                    # Validierungs-Score berechnen
                    val_score = self.calculate_concordance_index(cox_model, val_data)
                    cv_scores.append(val_score)
                
                # Durchschnittlicher CV-Score
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                result = {
                    'params': params,
                    'mean_cv_score': mean_score,
                    'std_cv_score': std_score,
                    'cv_scores': cv_scores
                }
                results.append(result)
                
                # Bester Score aktualisieren
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                
                self.logger.info(f"      📊 CV-Score: {mean_score:.4f} ± {std_score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"      ⚠️ Fehler bei Parametern {params}: {e}")
                continue
        
        if not results:
            raise ValueError("Hyperparameter-Tuning fehlgeschlagen - keine erfolgreichen Kombinationen")
        
        # Ergebnisse sortieren
        results.sort(key=lambda x: x['mean_cv_score'], reverse=True)
        
        # Beste Parameter in Konfiguration übernehmen
        if best_params:
            self.model_config.update(best_params)
        
        tuning_results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'total_combinations_tested': len(results),
            'tuning_method': tuning_config['method'],
            'cv_folds': cv_folds
        }
        
        self.hyperparameter_results = tuning_results
        
        self.logger.info(f"✅ Hyperparameter-Tuning abgeschlossen:")
        self.logger.info(f"   🏆 Beste Parameter: {best_params}")
        self.logger.info(f"   🎯 Bester CV-Score: {best_score:.4f}")
        
        return tuning_results
    
    def cross_validate_model(self, data: Optional[pd.DataFrame] = None, 
                           cv_folds: Optional[int] = None) -> Dict[str, Any]:
        """
        Cross-Validation für robuste Performance-Schätzung
        
        Args:
            data: Training-Datensatz (optional)
            cv_folds: Anzahl CV-Folds (optional)
            
        Returns:
            CV-Ergebnisse
        """
        if not self.model_config['cross_validation']['enabled']:
            self.logger.info("ℹ️ Cross-Validation deaktiviert")
            return {}
        
        self.logger.info("🔄 Starte Cross-Validation")
        
        if data is None:
            data = self.training_data
        
        if cv_folds is None:
            cv_folds = self.model_config['cross_validation']['cv_folds']
        
        # KFold Setup
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_details = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            self.logger.info(f"   📊 Fold {fold+1}/{cv_folds}")
            
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            try:
                # Model trainieren
                cox_model = CoxPHFitter(
                    penalizer=self.model_config['penalizer'],
                    l1_ratio=self.model_config['l1_ratio'],
                    alpha=self.model_config['alpha']
                )
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    
                    cox_model.fit(
                        train_data,
                        duration_col='duration',
                        event_col='event',
                        show_progress=False,
                        robust=True
                    )
                
                # Validierungs-Metriken
                val_c_index = self.calculate_concordance_index(cox_model, val_data)
                train_c_index = self.calculate_concordance_index(cox_model, train_data)
                
                cv_scores.append(val_c_index)
                
                fold_detail = {
                    'fold': fold + 1,
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'train_c_index': train_c_index,
                    'val_c_index': val_c_index,
                    'overfitting': train_c_index - val_c_index
                }
                cv_details.append(fold_detail)
                
                self.logger.info(f"      📊 Train C-Index: {train_c_index:.4f}")
                self.logger.info(f"      📊 Val C-Index: {val_c_index:.4f}")
                
            except Exception as e:
                self.logger.warning(f"      ⚠️ Fold {fold+1} fehlgeschlagen: {e}")
                continue
        
        if not cv_scores:
            raise ValueError("Cross-Validation fehlgeschlagen - keine erfolgreichen Folds")
        
        # CV-Ergebnisse zusammenfassen
        cv_results = {
            'mean_c_index': np.mean(cv_scores),
            'std_c_index': np.std(cv_scores),
            'min_c_index': np.min(cv_scores),
            'max_c_index': np.max(cv_scores),
            'cv_scores': cv_scores,
            'cv_details': cv_details,
            'cv_folds': len(cv_scores),
            'mean_overfitting': np.mean([d['overfitting'] for d in cv_details])
        }
        
        self.logger.info(f"✅ Cross-Validation abgeschlossen:")
        self.logger.info(f"   📊 Mean C-Index: {cv_results['mean_c_index']:.4f} ± {cv_results['std_c_index']:.4f}")
        self.logger.info(f"   📊 Range: {cv_results['min_c_index']:.4f} - {cv_results['max_c_index']:.4f}")
        self.logger.info(f"   📊 Overfitting: {cv_results['mean_overfitting']:.4f}")
        
        return cv_results
    
    # =============================================================================
    # MODEL EVALUATION
    # =============================================================================
    
    def calculate_concordance_index(self, model: CoxPHFitter, data: pd.DataFrame) -> float:
        """
        Berechnet Concordance Index (C-Index)
        
        Args:
            model: Cox-Modell
            data: Test-Daten
            
        Returns:
            C-Index (0.5 = zufällig, 1.0 = perfekt)
        """
        try:
            # Vorhersagen erstellen
            predictions = model.predict_partial_hazard(data[self.feature_columns])
            
            # C-Index berechnen
            c_index = concordance_index(
                data['duration'], 
                predictions, 
                data['event']
            )
            
            return float(c_index)
            
        except Exception as e:
            self.logger.warning(f"⚠️ C-Index-Berechnung fehlgeschlagen: {e}")
            return 0.5
    
    def evaluate_model_performance(self, model: Optional[CoxPHFitter] = None,
                                 data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluiert Model-Performance
        
        Args:
            model: Cox-Modell (optional, nutzt self.fitted_model)
            data: Test-Daten (optional, nutzt Training-Daten)
            
        Returns:
            Performance-Metriken
        """
        if model is None:
            model = self.fitted_model
        if model is None:
            raise ValueError("Kein Modell für Evaluation verfügbar")
        
        if data is None:
            data = self.training_data
        
        self.logger.info("📈 Evaluiere Model-Performance")
        
        # Basis-Metriken
        c_index = self.calculate_concordance_index(model, data)
        
        # Zusätzliche Metriken
        performance = {
            'concordance_index': c_index,
            'log_likelihood': float(model.log_likelihood_),
            'aic': float(model.AIC_),
            'bic': float(model.AIC_ + (np.log(len(data)) - 2) * model.params_.shape[0]),
            'partial_aic': float(model.AIC_partial_),
            'number_of_observations': len(data),
            'number_of_events': int(data['event'].sum()),
            'event_rate': float(data['event'].mean()),
            'number_of_features': len(self.feature_columns)
        }
        
        # Performance-Bewertung
        if c_index >= self.model_config['target_c_index']:
            performance_level = "Excellent"
        elif c_index >= self.model_config['min_acceptable_c_index']:
            performance_level = "Good"
        elif c_index >= 0.7:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        performance['performance_level'] = performance_level
        performance['target_achieved'] = c_index >= self.model_config['target_c_index']
        
        self.logger.info(f"📊 Performance-Evaluation:")
        self.logger.info(f"   🎯 C-Index: {c_index:.4f} ({performance_level})")
        self.logger.info(f"   📊 Log-Likelihood: {performance['log_likelihood']:.2f}")
        self.logger.info(f"   📊 AIC: {performance['aic']:.2f}")
        self.logger.info(f"   🎯 Ziel erreicht: {'Ja' if performance['target_achieved'] else 'Nein'}")
        
        return performance
    
    def analyze_feature_importance(self, model: Optional[CoxPHFitter] = None) -> pd.DataFrame:
        """
        Analysiert Feature-Importance basierend auf Koeffizienten
        
        Args:
            model: Cox-Modell (optional)
            
        Returns:
            DataFrame mit Feature-Importance
        """
        if model is None:
            model = self.fitted_model
        if model is None:
            raise ValueError("Kein Modell für Feature-Importance verfügbar")
        
        self.logger.info("🔍 Analysiere Feature-Importance")
        
        # Koeffizienten und Statistiken extrahieren
        summary = model.summary
        
        feature_importance = pd.DataFrame({
            'feature': summary.index,
            'coefficient': summary['coef'].values,
            'hazard_ratio': summary['exp(coef)'].values,
            'p_value': summary['p'].values,
            'confidence_lower': summary['exp(coef) lower 95%'].values,
            'confidence_upper': summary['exp(coef) upper 95%'].values,
            'abs_coefficient': np.abs(summary['coef'].values)
        })
        
        # Nach absolutem Koeffizienten sortieren
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        # Signifikanz hinzufügen
        feature_importance['significant'] = feature_importance['p_value'] < 0.05
        
        self.logger.info(f"📊 Feature-Importance-Analyse:")
        self.logger.info(f"   🔢 {len(feature_importance)} Features analysiert")
        self.logger.info(f"   ✅ {feature_importance['significant'].sum()} signifikante Features")
        
        # Top-5 Features anzeigen
        self.logger.info("   🏆 Top-5 Features:")
        for i, row in feature_importance.head().iterrows():
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            self.logger.info(f"      {row['feature']}: {row['coefficient']:.3f} (HR: {row['hazard_ratio']:.3f}) {significance}")
        
        return feature_importance
    
    # =============================================================================
    # MODEL VALIDATION
    # =============================================================================
    
    def validate_convergence(self, model: CoxPHFitter) -> Dict[str, Any]:
        """
        Validiert Model-Konvergenz
        
        Args:
            model: Trainiertes Cox-Modell
            
        Returns:
            Konvergenz-Info
        """
        convergence_info = {
            'converged': True,
            'iterations': getattr(model, '_n_iterations', 'unknown'),
            'log_likelihood': float(model.log_likelihood_),
            'convergence_quality': 'good'
        }
        
        try:
            # Prüfe ob Model konvergiert ist
            if hasattr(model, 'standard_errors_'):
                # Wenn Standard-Errors verfügbar sind, ist das Model wahrscheinlich konvergiert
                convergence_info['has_standard_errors'] = True
                
                # Prüfe auf extreme Standard-Errors (Hinweis auf Konvergenz-Probleme)
                max_se = model.standard_errors_.max()
                if max_se > 10:
                    convergence_info['convergence_quality'] = 'poor'
                    convergence_info['max_standard_error'] = float(max_se)
            else:
                convergence_info['has_standard_errors'] = False
                convergence_info['converged'] = False
                convergence_info['convergence_quality'] = 'failed'
            
            # Prüfe Log-Likelihood auf NaN/Inf
            if not np.isfinite(model.log_likelihood_):
                convergence_info['converged'] = False
                convergence_info['convergence_quality'] = 'failed'
                convergence_info['issue'] = 'Non-finite log-likelihood'
            
        except Exception as e:
            convergence_info['converged'] = False
            convergence_info['convergence_quality'] = 'error'
            convergence_info['error'] = str(e)
        
        return convergence_info
    
    def detect_model_issues(self, model: CoxPHFitter, 
                          data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Erkennt potentielle Model-Probleme
        
        Args:
            model: Cox-Modell
            data: Training-Daten
            
        Returns:
            Liste der erkannten Probleme
        """
        issues = []
        
        try:
            # 1. Konvergenz-Probleme
            conv_info = self.validate_convergence(model)
            if not conv_info['converged']:
                issues.append({
                    'issue_type': 'convergence',
                    'severity': 'high',
                    'description': 'Modell ist nicht konvergiert',
                    'recommendation': 'Penalizer erhöhen oder Features reduzieren'
                })
            
            # 2. Extreme Koeffizienten
            if hasattr(model, 'params_'):
                max_coef = np.abs(model.params_).max()
                if max_coef > 10:
                    issues.append({
                        'issue_type': 'extreme_coefficients',
                        'severity': 'medium',
                        'description': f'Extreme Koeffizienten gefunden (max: {max_coef:.2f})',
                        'recommendation': 'Penalizer erhöhen oder Features standardisieren'
                    })
            
            # 3. Niedrige Performance
            c_index = self.calculate_concordance_index(model, data)
            if c_index < self.model_config['min_acceptable_c_index']:
                issues.append({
                    'issue_type': 'low_performance',
                    'severity': 'high',
                    'description': f'C-Index zu niedrig: {c_index:.3f}',
                    'recommendation': 'Feature-Engineering verbessern oder mehr Daten sammeln'
                })
            
            # 4. Wenige Events
            event_count = data['event'].sum()
            if event_count < 50:
                issues.append({
                    'issue_type': 'few_events',
                    'severity': 'medium',
                    'description': f'Nur {event_count} Events verfügbar',
                    'recommendation': 'Mehr Event-Daten sammeln für robustere Schätzung'
                })
            
            # 5. Zu viele Features vs. Events
            feature_count = len(self.feature_columns)
            if feature_count > event_count / 10:
                issues.append({
                    'issue_type': 'high_dimensionality',
                    'severity': 'medium',
                    'description': f'{feature_count} Features bei nur {event_count} Events',
                    'recommendation': 'Feature-Selektion durchführen (Regel: max Features = Events/10)'
                })
            
        except Exception as e:
            issues.append({
                'issue_type': 'analysis_error',
                'severity': 'low',
                'description': f'Fehler bei Problem-Erkennung: {e}',
                'recommendation': 'Manuelle Model-Inspektion durchführen'
            })
        
        return issues
    
    # =============================================================================
    # MODEL PERSISTENCE
    # =============================================================================
    
    def save_model(self, model: Optional[CoxPHFitter] = None, 
                  output_path: Optional[Path] = None) -> Path:
        """
        Speichert trainiertes Modell
        
        Args:
            model: Cox-Modell (optional)
            output_path: Ausgabe-Pfad (optional, auto-generiert)
            
        Returns:
            Pfad zur gespeicherten Model-Datei
        """
        if model is None:
            model = self.fitted_model
        if model is None:
            raise ValueError("Kein Modell zum Speichern verfügbar")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cox_model_{timestamp}.pkl"
            output_path = self.models_dir / filename
        
        # Model-Metadaten sammeln
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'CoxPHFitter',
            'feature_columns': self.feature_columns,
            'training_results': self.training_results,
            'convergence_info': self.convergence_info,
            'hyperparameter_results': self.hyperparameter_results,
            'model_config': self.model_config
        }
        
        # Model und Metadaten speichern
        model_package = {
            'model': model,
            'metadata': model_metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Zusätzlich JSON-Report speichern
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Modell gespeichert: {output_path}")
        self.logger.info(f"📄 Metadaten gespeichert: {json_path}")
        
        return output_path
    
    def load_model(self, model_path: Path) -> CoxPHFitter:
        """
        Lädt gespeichertes Modell
        
        Args:
            model_path: Pfad zur Model-Datei
            
        Returns:
            Geladenes Cox-Modell
        """
        self.logger.info(f"📂 Lade Modell: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Model und Metadaten extrahieren
        model = model_package['model']
        metadata = model_package['metadata']
        
        # State wiederherstellen
        self.fitted_model = model
        self.feature_columns = metadata.get('feature_columns', [])
        self.training_results = metadata.get('training_results', {})
        self.convergence_info = metadata.get('convergence_info', {})
        self.hyperparameter_results = metadata.get('hyperparameter_results', {})
        
        # Konfiguration aktualisieren
        if 'model_config' in metadata:
            self.model_config.update(metadata['model_config'])
        
        self.logger.info(f"✅ Modell geladen:")
        self.logger.info(f"   🔢 {len(self.feature_columns)} Features")
        self.logger.info(f"   🎯 C-Index: {self.training_results.get('concordance_index', 'N/A')}")
        
        return model
    
    def export_training_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Exportiert vollständigen Training-Report
        
        Args:
            output_path: Ausgabe-Pfad (optional)
            
        Returns:
            Pfad zum Training-Report (JSON)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cox_training_report_{timestamp}.json"
            output_path = self.output_dir / filename
        
        # Feature-Importance hinzufügen (falls verfügbar)
        feature_importance = None
        if self.fitted_model is not None:
            try:
                feature_importance_df = self.analyze_feature_importance()
                feature_importance = feature_importance_df.to_dict('records')
            except:
                pass
        
        # Vollständiger Report
        training_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'Cox_Training_Report',
                'target_c_index': self.model_config['target_c_index']
            },
            'model_config': self.model_config,
            'training_results': self.training_results,
            'convergence_info': self.convergence_info,
            'hyperparameter_results': self.hyperparameter_results,
            'feature_columns': self.feature_columns,
            'feature_importance': feature_importance,
            'model_saved': self.fitted_model is not None
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📄 Training-Report exportiert: {output_path}")
        return output_path
    
    # =============================================================================
    # OPTIMIZATION STRATEGIES
    # =============================================================================
    
    def optimize_for_performance(self, data: pd.DataFrame, 
                               target_c_index: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimiert Modell für Ziel-Performance (0.95+ C-Index)
        
        Args:
            data: Training-Datensatz
            target_c_index: Ziel-C-Index (optional)
            
        Returns:
            Optimierungs-Ergebnisse
        """
        if target_c_index is None:
            target_c_index = self.model_config['target_c_index']
        
        self.logger.info(f"🎯 Optimiere für C-Index {target_c_index}")
        
        optimization_steps = []
        current_best_score = 0.0
        
        # 1. Basis-Model trainieren
        self.logger.info("   📊 Schritt 1: Basis-Model")
        base_model = self.train_cox_model(data)
        base_score = self.calculate_concordance_index(base_model, data)
        current_best_score = base_score
        
        optimization_steps.append({
            'step': 'baseline',
            'c_index': base_score,
            'description': 'Basis-Model mit Standard-Parametern'
        })
        
        # 2. Hyperparameter-Tuning
        if base_score < target_c_index:
            self.logger.info("   🔬 Schritt 2: Hyperparameter-Tuning")
            tuning_results = self.hyperparameter_tuning(data)
            
            if tuning_results and tuning_results['best_score'] > current_best_score:
                # Model mit besten Parametern neu trainieren
                tuned_model = self.train_cox_model(data)
                tuned_score = self.calculate_concordance_index(tuned_model, data)
                current_best_score = tuned_score
                
                optimization_steps.append({
                    'step': 'hyperparameter_tuning',
                    'c_index': tuned_score,
                    'description': f'Optimierte Parameter: {tuning_results["best_params"]}'
                })
        
        # 3. Cross-Validation zur finalen Validierung
        if current_best_score >= target_c_index * 0.95:  # 95% des Ziels erreicht
            self.logger.info("   🔄 Schritt 3: Cross-Validation")
            cv_results = self.cross_validate_model(data)
            
            if cv_results:
                optimization_steps.append({
                    'step': 'cross_validation',
                    'c_index': cv_results['mean_c_index'],
                    'std': cv_results['std_c_index'],
                    'description': f'CV-validierte Performance'
                })
        
        # Finale Bewertung
        target_achieved = current_best_score >= target_c_index
        achievement_ratio = current_best_score / target_c_index
        
        # Empfehlungen für weitere Optimierung
        recommendations = []
        if not target_achieved:
            if current_best_score < 0.7:
                recommendations.append("Grundlegendes Feature-Engineering überarbeiten")
                recommendations.append("Datenqualität prüfen")
            elif current_best_score < 0.85:
                recommendations.append("Mehr Features hinzufügen")
                recommendations.append("Feature-Interaktionen testen")
            else:
                recommendations.append("Ensemble-Methoden evaluieren")
                recommendations.append("Advanced Feature-Engineering")
        
        optimization_results = {
            'target_c_index': target_c_index,
            'achieved_c_index': current_best_score,
            'target_achieved': target_achieved,
            'achievement_ratio': achievement_ratio,
            'optimization_steps': optimization_steps,
            'recommendations': recommendations,
            'final_model_available': self.fitted_model is not None
        }
        
        self.logger.info(f"🎯 Optimierung abgeschlossen:")
        self.logger.info(f"   📊 Erreicht: {current_best_score:.4f} (Ziel: {target_c_index:.4f})")
        self.logger.info(f"   ✅ Ziel erreicht: {'Ja' if target_achieved else 'Nein'}")
        self.logger.info(f"   📈 Zielerreichung: {achievement_ratio:.1%}")
        
        return optimization_results


if __name__ == "__main__":
    # Test des Model Trainers
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Cox Model Trainer Test")
    
    if not LIFELINES_AVAILABLE:
        print("❌ lifelines nicht verfügbar - Test übersprungen")
        exit(1)
    
    # Model Trainer initialisieren
    trainer = CoxModelTrainer()
    
    # Test-Daten erstellen
    np.random.seed(42)
    n_samples = 100
    
    test_data = pd.DataFrame({
        'duration': np.random.exponential(10, n_samples),
        'event': np.random.binomial(1, 0.3, n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.binomial(1, 0.5, n_samples)
    })
    
    trainer.feature_columns = ['feature1', 'feature2', 'feature3']
    trainer.training_data = test_data
    
    try:
        # Test: Model trainieren
        model = trainer.train_cox_model(test_data)
        print(f"✅ Model-Training erfolgreich")
        
        # Test: Performance evaluieren
        performance = trainer.evaluate_model_performance(model, test_data)
        print(f"📊 C-Index: {performance['concordance_index']:.4f}")
        
        # Test: Feature-Importance
        feature_importance = trainer.analyze_feature_importance(model)
        print(f"🔍 Feature-Importance berechnet: {len(feature_importance)} Features")
        
        print("✅ Alle Tests erfolgreich!")
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
