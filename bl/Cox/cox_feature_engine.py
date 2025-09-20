#!/usr/bin/env python3
"""
Cox Feature Engine - Konsolidiertes Feature-Engineering für Cox-Analyse
========================================================================

Konsolidiert alle 3 Feature-Engineering-Ansätze:
1. Rolling-Features aus cox_optimized_analyzer.py (bewährt: 0.890 C-Index)
2. Enhanced Features aus cox_enhanced_features_integration.py (110+ Features)
3. Data Dictionary Integration aus cox_feature_manager.py

Ziel: 0.95+ C-Index durch optimierte Feature-Selektion

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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import VarianceThreshold
import warnings

# Projekt-Pfade hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.paths_config import ProjectPaths
    from bl.Cox.cox_constants import (
        HUNDRED_PERCENT, SIX_MONTH_HORIZON, TWELVE_MONTH_HORIZON,
        HIGH_CORRELATION_THRESHOLD, NEAR_ZERO_VARIANCE_THRESHOLD
    )
except ImportError as e:
    print(f"⚠️ Import-Fehler: {e}")
    # Fallback-Konstanten
    HUNDRED_PERCENT = 100
    SIX_MONTH_HORIZON = 6
    TWELVE_MONTH_HORIZON = 12
    HIGH_CORRELATION_THRESHOLD = 0.95
    NEAR_ZERO_VARIANCE_THRESHOLD = 0.01


class CoxFeatureEngine:
    """
    Konsolidiertes Feature-Engineering für Cox-Analyse
    Kombiniert beste Ansätze aus allen 3 bestehenden Implementierungen
    """
    
    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert Feature Engine
        
        Args:
            feature_config: Konfiguration für Feature-Engineering
        """
        self.feature_config = feature_config or self._default_config()
        self.logger = self._setup_logging()
        
        # Paths
        try:
            self.paths = ProjectPaths()
        except:
            self.paths = None
            
        # State
        self.data_dictionary: Optional[Dict[str, Any]] = None
        self.enhanced_features: Optional[List[str]] = None
        self.selected_features: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.feature_importance: Dict[str, float] = {}
        self.feature_stats: Dict[str, Any] = {}
        
        self.logger.info("🔧 Cox Feature Engine initialisiert")
    
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
    # CONFIGURATION & SETUP
    # =============================================================================
    
    def _default_config(self) -> Dict[str, Any]:
        """Standard-Konfiguration für Feature-Engineering"""
        return {
            'rolling_windows': [12],  # Minimal: nur 12-Monats-Fenster für I_SOCIALINSURANCENOTES
            'loopback_months': 6,  # Monate Abstand zum Event (gegen Data Leakage)
            'use_enhanced_features': False,   # Temporär deaktiviert - fokus auf Basis-Features
            'use_data_dictionary': True,      # Data Dictionary Integration
            'feature_selection_k': 15,        # Top-K Features
            'standardize_features': False,    # Deaktiviert wegen Division-durch-Null bei wenig Varianz
            'remove_high_correlation': True,  # Korrelations-Filter
            'correlation_threshold': 0.6,  # Ultra-aggressiv gegen Korrelation - Data Leakage Schutz
            'variance_threshold': 0.1,  # Erhöht gegen Near-Zero-Variance bei zensierten Daten
            'base_features': ['I_SOCIALINSURANCENOTES'],  # HAUPTFEATURE: Unabhängig von I_Alive Status
            'categorical_features': ['N_DIGITALIZATIONRATE'],  # One-Hot Features
            'enhanced_categories': [
                'activity_features',
                'rolling_features', 
                'trend_features',
                'early_warning_features',
                'categorical_features',
                'interaction_features'
            ]
        }
    
    def load_data_dictionary(self, dict_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Lädt Data Dictionary für Feature-Konfiguration
        
        Args:
            dict_path: Pfad zum Data Dictionary (optional)
            
        Returns:
            Data Dictionary Inhalt
        """
        try:
            if dict_path is None and self.paths:
                dict_path = self.paths.config_directory() / "data_dictionary_optimized.json"
            elif dict_path is None:
                dict_path = Path("config/data_dictionary_optimized.json")
            
            self.logger.info(f"📊 Lade Data Dictionary: {dict_path}")
            
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.data_dictionary = json.load(f)
            
            self.logger.info(f"✅ Data Dictionary geladen ({len(self.data_dictionary.get('columns', {}))} Spalten)")
            return self.data_dictionary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data Dictionary konnte nicht geladen werden: {e}")
            self.data_dictionary = {}
            return {}
    
    def load_enhanced_features_config(self, enhanced_path: Optional[Path] = None) -> List[str]:
        """
        Lädt Enhanced Features Konfiguration
        
        Args:
            enhanced_path: Pfad zur enhanced_features.json (optional)
            
        Returns:
            Liste der Enhanced Feature-Namen
        """
        try:
            if enhanced_path is None and self.paths:
                stage1_dir = self.paths.dynamic_outputs_directory() / "stage1_outputs"
                enhanced_path = stage1_dir / "enhanced_features.json"
            elif enhanced_path is None:
                enhanced_path = Path("dynamic_system_outputs/stage1_outputs/enhanced_features.json")
            
            if not enhanced_path.exists():
                self.logger.warning(f"⚠️ Enhanced Features Datei nicht gefunden: {enhanced_path}")
                return []
            
            self.logger.info(f"📊 Lade Enhanced Features: {enhanced_path}")
            
            with open(enhanced_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)
            
            # Feature-Namen extrahieren
            if 'enhanced_features' in enhanced_data and enhanced_data['enhanced_features']:
                first_record = enhanced_data['enhanced_features'][0]
                feature_names = [col for col in first_record.keys() 
                               if col not in ['Kunde', 'I_TIMEBASE']]
                
                self.enhanced_features = feature_names
                self.logger.info(f"✅ {len(feature_names)} Enhanced Features geladen")
                return feature_names
            else:
                self.logger.warning("⚠️ Keine Enhanced Features in der JSON-Datei gefunden")
                return []
                
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced Features konnten nicht geladen werden: {e}")
            return []
    
    # =============================================================================
    # CORE FEATURE ENGINEERING (aus cox_optimized_analyzer.py)
    # =============================================================================
    
    def create_rolling_features_with_loopback(self, data: pd.DataFrame, survival_panel: pd.DataFrame,
                              base_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Erstellt Rolling-Features mit Loopback-Logik (gegen Data Leakage)
        
        Args:
            data: Stage0-Daten
            survival_panel: Cox-Survival-Panel mit Event-Zeiten
            base_features: Basis-Features für Rolling
            
        Returns:
            DataFrame mit Rolling-Features (ohne Data Leakage)
        """
        self.logger.info("🔄 Erstelle Rolling-Features mit Loopback")
        
        if base_features is None:
            base_features = self.feature_config['base_features']
        
        windows = self.feature_config['rolling_windows']
        loopback_months = self.feature_config['loopback_months']
        rolling_features_df = survival_panel[['Kunde']].copy()
        
        self.logger.info(f"   🔒 Loopback: {loopback_months} Monate vor Event")
        
        for feature in base_features:
            if feature not in data.columns:
                self.logger.warning(f"⚠️ Feature {feature} nicht in Daten gefunden")
                continue
                
            self.logger.info(f"  📊 Verarbeite {feature} mit Loopback")
            
            # Pro Kunde Rolling-Features mit Loopback erstellen
            for _, survival_row in survival_panel.iterrows():
                kunde = survival_row['Kunde']
                
                # Event-Zeitpunkt aus Survival-Panel
                if 'last_observed' in survival_row:
                    event_timebase = survival_row['last_observed']
                else:
                    # Fallback: Nutze t_end
                    global_start = data['I_TIMEBASE'].min()
                    event_timebase = global_start + survival_row['t_end']
                
                # Cutoff-Zeitpunkt für Features (loopback_months vor Event)
                cutoff_timebase = event_timebase - loopback_months
                
                # Kundendaten bis Cutoff
                kunde_data = data[
                    (data['Kunde'] == kunde) & 
                    (data['I_TIMEBASE'] <= cutoff_timebase)
                ].sort_values('I_TIMEBASE')
                
                if len(kunde_data) < max(windows):
                    # Nicht genug historische Daten
                    continue
                
                for window in windows:
                    # Rolling-Features berechnen
                    rolling_mean = kunde_data[feature].rolling(window=window, min_periods=1).mean()
                    rolling_sum = kunde_data[feature].rolling(window=window, min_periods=1).sum()
                    rolling_std = kunde_data[feature].rolling(window=window, min_periods=1).std()
                    
                    # Activity Rate
                    activity_rate = kunde_data[feature].rolling(window=window, min_periods=1).apply(
                        lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
                    )
                    
                    # Trend
                    def calculate_trend(series):
                        if len(series) < 2:
                            return 0
                        x = np.arange(len(series))
                        try:
                            return np.polyfit(x, series, 1)[0]
                        except:
                            return 0
                    
                    trend = kunde_data[feature].rolling(window=window, min_periods=2).apply(calculate_trend)
                    
                    # Letzten verfügbaren Wert nehmen (zum Cutoff-Zeitpunkt)
                    if len(rolling_mean) > 0:
                        mean_val = rolling_mean.iloc[-1]
                        sum_val = rolling_sum.iloc[-1]
                        std_val = rolling_std.iloc[-1] if not pd.isna(rolling_std.iloc[-1]) else 0
                        activity_val = activity_rate.iloc[-1]
                        trend_val = trend.iloc[-1] if len(trend) > 0 and not pd.isna(trend.iloc[-1]) else 0
                        
                        # Feature-Namen
                        mean_col = f"{feature}_rolling_{window}p_mean"
                        sum_col = f"{feature}_rolling_{window}p_sum"
                        std_col = f"{feature}_rolling_{window}p_std"
                        activity_col = f"{feature}_activity_rate_{window}p"
                        trend_col = f"{feature}_trend_{window}p"
                        
                        # In DataFrame eintragen
                        kunde_idx = rolling_features_df[rolling_features_df['Kunde'] == kunde].index
                        
                        if len(kunde_idx) > 0:
                            rolling_features_df.loc[kunde_idx[0], mean_col] = mean_val
                            rolling_features_df.loc[kunde_idx[0], sum_col] = sum_val
                            rolling_features_df.loc[kunde_idx[0], std_col] = std_val
                            rolling_features_df.loc[kunde_idx[0], activity_col] = activity_val
                            rolling_features_df.loc[kunde_idx[0], trend_col] = trend_val
        
        # NaN-Werte mit 0 füllen
        feature_cols = [col for col in rolling_features_df.columns if col != 'Kunde']
        rolling_features_df[feature_cols] = rolling_features_df[feature_cols].fillna(0)
        
        self.logger.info(f"✅ {len(feature_cols)} Rolling-Features mit Loopback erstellt")
        return rolling_features_df

    def create_rolling_features(self, data: pd.DataFrame, 
                              base_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Erstellt Rolling-Features (bewährt: 0.890 C-Index)
        
        Args:
            data: Input-Daten (Stage0 oder Panel)
            base_features: Basis-Features für Rolling (default: I_MAINTENANCE)
            
        Returns:
            DataFrame mit Rolling-Features:
            - [FEATURE]_rolling_[WINDOW]p_mean
            - [FEATURE]_rolling_[WINDOW]p_sum
            - [FEATURE]_rolling_[WINDOW]p_std
            - [FEATURE]_activity_rate_[WINDOW]p
            - [FEATURE]_trend_[WINDOW]p
        """
        self.logger.info("🔄 Erstelle Rolling-Features")
        
        if base_features is None:
            base_features = self.feature_config['base_features']
        
        windows = self.feature_config['rolling_windows']
        rolling_features_df = data[['Kunde']].copy()
        
        for feature in base_features:
            if feature not in data.columns:
                self.logger.warning(f"⚠️ Feature {feature} nicht in Daten gefunden")
                continue
                
            self.logger.info(f"  📊 Verarbeite {feature}")
            
            # Pro Kunde Rolling-Features erstellen
            for kunde in data['Kunde'].unique():
                kunde_data = data[data['Kunde'] == kunde].sort_values('I_TIMEBASE')
                
                if len(kunde_data) < max(windows):
                    continue  # Nicht genug Daten für größtes Fenster
                
                for window in windows:
                    # Rolling Mean
                    rolling_mean = kunde_data[feature].rolling(window=window, min_periods=1).mean()
                    mean_col = f"{feature}_rolling_{window}p_mean"
                    
                    # Rolling Sum  
                    rolling_sum = kunde_data[feature].rolling(window=window, min_periods=1).sum()
                    sum_col = f"{feature}_rolling_{window}p_sum"
                    
                    # Rolling Std
                    rolling_std = kunde_data[feature].rolling(window=window, min_periods=1).std()
                    std_col = f"{feature}_rolling_{window}p_std"
                    
                    # Activity Rate (Anteil der Monate mit Feature > 0)
                    activity_rate = kunde_data[feature].rolling(window=window, min_periods=1).apply(
                        lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
                    )
                    activity_col = f"{feature}_activity_rate_{window}p"
                    
                    # Trend (Slope der letzten window Werte)
                    def calculate_trend(series):
                        if len(series) < 2:
                            return 0
                        x = np.arange(len(series))
                        try:
                            slope = np.polyfit(x, series, 1)[0]
                            return slope
                        except:
                            return 0
                    
                    trend = kunde_data[feature].rolling(window=window, min_periods=2).apply(calculate_trend)
                    trend_col = f"{feature}_trend_{window}p"
                    
                    # Letzten Wert pro Kunde nehmen (für Cox-Panel)
                    last_idx = kunde_data.index[-1]
                    kunde_idx = rolling_features_df[rolling_features_df['Kunde'] == kunde].index
                    
                    if len(kunde_idx) == 0:
                        # Kunde hinzufügen
                        new_row = {'Kunde': kunde}
                        rolling_features_df = pd.concat([rolling_features_df, pd.DataFrame([new_row])], ignore_index=True)
                        kunde_idx = rolling_features_df[rolling_features_df['Kunde'] == kunde].index
                    
                    rolling_features_df.loc[kunde_idx, mean_col] = rolling_mean.iloc[-1]
                    rolling_features_df.loc[kunde_idx, sum_col] = rolling_sum.iloc[-1]
                    rolling_features_df.loc[kunde_idx, std_col] = rolling_std.iloc[-1]
                    rolling_features_df.loc[kunde_idx, activity_col] = activity_rate.iloc[-1]
                    rolling_features_df.loc[kunde_idx, trend_col] = trend.iloc[-1]
        
        # NaN-Werte durch 0 ersetzen
        feature_cols = [col for col in rolling_features_df.columns if col != 'Kunde']
        rolling_features_df[feature_cols] = rolling_features_df[feature_cols].fillna(0)
        
        self.logger.info(f"✅ {len(feature_cols)} Rolling-Features erstellt")
        return rolling_features_df
    
    def create_onehot_features(self, data: pd.DataFrame, 
                             categorical_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Erstellt One-Hot-Encoded Features (bewährt für N_DIGITALIZATIONRATE)
        
        Args:
            data: Input-Daten
            categorical_features: Kategorische Features (default: N_DIGITALIZATIONRATE)
            
        Returns:
            DataFrame mit One-Hot Features:
            - [FEATURE]_[VALUE] (z.B. N_DIGITALIZATIONRATE_2.0)
        """
        self.logger.info("🏷️ Erstelle One-Hot Features")
        
        if categorical_features is None:
            categorical_features = self.feature_config['categorical_features']
        
        onehot_df = data[['Kunde']].copy()
        
        for feature in categorical_features:
            if feature not in data.columns:
                self.logger.warning(f"⚠️ Feature {feature} nicht in Daten gefunden")
                continue
                
            self.logger.info(f"  🏷️ Verarbeite {feature}")
            
            # Pro Kunde letzten Wert nehmen
            kunde_feature_data = data.groupby('Kunde')[feature].last().reset_index()
            
            # One-Hot Encoding
            onehot_encoded = pd.get_dummies(kunde_feature_data[feature], prefix=feature, drop_first=False)
            
            # Mit Kunden-IDs kombinieren
            kunde_onehot = pd.concat([kunde_feature_data[['Kunde']], onehot_encoded], axis=1)
            
            # Mit Haupt-DataFrame mergen
            onehot_df = onehot_df.merge(kunde_onehot, on='Kunde', how='left')
        
        # NaN-Werte durch 0 ersetzen (für fehlende Kategorien)
        feature_cols = [col for col in onehot_df.columns if col != 'Kunde']
        onehot_df[feature_cols] = onehot_df[feature_cols].fillna(0)
        
        self.logger.info(f"✅ {len(feature_cols)} One-Hot Features erstellt")
        return onehot_df
    
    def create_activity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt Activity-Rate Features (wichtigste aus 0.890 Performance)
        
        Args:
            data: Input-Daten
            
        Returns:
            DataFrame mit Activity-Features:
            - I_MAINTENANCE_activity_rate_6p  (-26.67 Coeff)
            - I_MAINTENANCE_activity_rate_12p (49.64 Coeff) 
            - I_MAINTENANCE_activity_rate_18p (-41.95 Coeff)
        """
        self.logger.info("⚡ Erstelle Activity-Features (Top-Performance)")
        
        # Diese spezifischen Features haben in cox_optimized_analyzer die beste Performance gezeigt
        target_features = [
            'I_MAINTENANCE_activity_rate_6p',
            'I_MAINTENANCE_activity_rate_12p', 
            'I_MAINTENANCE_activity_rate_18p'
        ]
        
        # Rolling-Features erstellen (enthält Activity-Features)
        rolling_df = self.create_rolling_features(data, base_features=['I_MAINTENANCE'])
        
        # Nur die Top-Activity-Features extrahieren
        activity_cols = ['Kunde'] + [col for col in rolling_df.columns if col in target_features]
        
        if len(activity_cols) > 1:  # Mehr als nur 'Kunde'
            activity_df = rolling_df[activity_cols].copy()
            self.logger.info(f"✅ {len(activity_cols)-1} Top-Activity Features extrahiert")
            return activity_df
        else:
            self.logger.warning("⚠️ Keine Activity-Features gefunden, erstelle leeres DataFrame")
            return pd.DataFrame({'Kunde': data['Kunde'].unique()})
    
    # =============================================================================
    # ENHANCED FEATURES INTEGRATION
    # =============================================================================
    
    def create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Integriert Enhanced Features aus ChurnPipeline (110+ Features)
        
        Args:
            data: Input-Daten
            
        Returns:
            DataFrame mit Enhanced Features:
            - Early Warning Features
            - Cumulative Features
            - Interaction Features
            - Trend Features
        """
        self.logger.info("⚡ Integriere Enhanced Features")
        
        # Enhanced Features laden falls noch nicht geschehen
        if self.enhanced_features is None:
            self.enhanced_features = self.load_enhanced_features_config()
        
        if not self.enhanced_features:
            self.logger.warning("⚠️ Keine Enhanced Features verfügbar")
            return pd.DataFrame({'Kunde': data['Kunde'].unique()})
        
        # Enhanced Features aus stage1_outputs laden
        try:
            if self.paths:
                enhanced_path = self.paths.dynamic_outputs_directory() / "stage1_outputs" / "enhanced_features.json"
            else:
                enhanced_path = Path("dynamic_system_outputs/stage1_outputs/enhanced_features.json")
            
            with open(enhanced_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)
            
            if 'enhanced_features' not in enhanced_data:
                self.logger.warning("⚠️ Keine enhanced_features in JSON gefunden")
                return pd.DataFrame({'Kunde': data['Kunde'].unique()})
            
            # Enhanced Features DataFrame erstellen
            enhanced_df = pd.DataFrame(enhanced_data['enhanced_features'])
            
            # Nur relevante Features auswählen (ohne Kunde, I_TIMEBASE)
            feature_cols = [col for col in enhanced_df.columns 
                          if col not in ['Kunde', 'I_TIMEBASE']]
            
            enhanced_features_final = enhanced_df[['Kunde'] + feature_cols].copy()
            
            self.logger.info(f"✅ {len(feature_cols)} Enhanced Features integriert")
            return enhanced_features_final
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced Features Integration fehlgeschlagen: {e}")
            return pd.DataFrame({'Kunde': data['Kunde'].unique()})
    
    def categorize_enhanced_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Kategorisiert Enhanced Features nach Typen
        
        Args:
            feature_names: Liste der Feature-Namen
            
        Returns:
            Dict mit kategorisierten Features
        """
        categories = {
            'activity_features': [],
            'rolling_features': [],
            'trend_features': [],
            'early_warning_features': [],
            'categorical_features': [],
            'interaction_features': [],
            'cumulative_features': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if 'activity_rate' in feature_lower or 'activity' in feature_lower:
                categories['activity_features'].append(feature)
            elif 'rolling' in feature_lower or '_mean' in feature_lower or '_sum' in feature_lower:
                categories['rolling_features'].append(feature)
            elif 'trend' in feature_lower or 'pct_change' in feature_lower or 'slope' in feature_lower:
                categories['trend_features'].append(feature)
            elif 'early_warning' in feature_lower or 'ew_' in feature_lower:
                categories['early_warning_features'].append(feature)
            elif 'onehot' in feature_lower or feature.count('_') > 2:  # Heuristik für One-Hot
                categories['categorical_features'].append(feature)
            elif 'interaction' in feature_lower or 'ratio' in feature_lower:
                categories['interaction_features'].append(feature)
            elif 'cumsum' in feature_lower or 'cummax' in feature_lower or 'cumulative' in feature_lower:
                categories['cumulative_features'].append(feature)
        
        return categories
    
    # =============================================================================
    # FEATURE SELECTION & OPTIMIZATION
    # =============================================================================
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, 
                           method: str = 'univariate') -> List[str]:
        """
        Selektiert beste Features für Cox-Modell
        
        Args:
            X: Feature-Matrix
            y: Target (duration für univariate selection)
            method: Selection-Methode ('univariate', 'correlation', 'combined')
            
        Returns:
            Liste der selektierten Feature-Namen
        """
        self.logger.info(f"🎯 Feature-Selektion ({method}) - K={self.feature_config['feature_selection_k']}")
        
        feature_cols = [col for col in X.columns if col != 'Kunde']
        X_features = X[feature_cols]
        
        if method == 'univariate':
            # Univariate Feature-Selection
            k = min(self.feature_config['feature_selection_k'], len(feature_cols))
            selector = SelectKBest(score_func=f_regression, k=k)
            
            try:
                selector.fit(X_features, y)
                selected_mask = selector.get_support()
                selected_features = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
                
                # Feature-Scores speichern
                scores = selector.scores_
                for i, feature in enumerate(feature_cols):
                    self.feature_importance[feature] = scores[i] if not np.isnan(scores[i]) else 0
                
                self.logger.info(f"✅ {len(selected_features)} Features selektiert (univariate)")
                return selected_features
                
            except Exception as e:
                self.logger.warning(f"⚠️ Univariate Selektion fehlgeschlagen: {e}")
                return feature_cols[:self.feature_config['feature_selection_k']]
        
        elif method == 'correlation':
            # Korrelations-basierte Selektion
            correlation_matrix = X_features.corr().abs()
            
            # Finde hoch-korrelierte Paare
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > self.feature_config['correlation_threshold']:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            # Entferne Features mit höchster durchschnittlicher Korrelation
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                mean_corr1 = correlation_matrix[feat1].mean()
                mean_corr2 = correlation_matrix[feat2].mean()
                if mean_corr1 > mean_corr2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            
            selected_features = [f for f in feature_cols if f not in features_to_remove]
            
            self.logger.info(f"✅ {len(selected_features)} Features nach Korrelations-Filter")
            return selected_features[:self.feature_config['feature_selection_k']]
        
        elif method == 'combined':
            # Kombinierte Selektion
            # 1. Erst Korrelations-Filter
            corr_features = self.select_best_features(X, y, method='correlation')
            
            # 2. Dann univariate Selektion auf gefilterte Features
            X_filtered = X[['Kunde'] + corr_features]
            final_features = self.select_best_features(X_filtered, y, method='univariate')
            
            self.logger.info(f"✅ {len(final_features)} Features nach kombinierter Selektion")
            return final_features
        
        else:
            self.logger.warning(f"⚠️ Unbekannte Selektions-Methode: {method}")
            return feature_cols[:self.feature_config['feature_selection_k']]
    
    def remove_correlated_features(self, data: pd.DataFrame, 
                                 threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Entfernt hoch-korrelierte Features
        
        Args:
            data: Feature-DataFrame
            threshold: Korrelations-Schwellwert
            
        Returns:
            DataFrame ohne hoch-korrelierte Features
        """
        if threshold is None:
            threshold = self.feature_config['correlation_threshold']
        
        self.logger.info(f"🔗 Entferne korrelierte Features (threshold={threshold})")
        
        feature_cols = [col for col in data.columns if col != 'Kunde']
        
        if len(feature_cols) < 2:
            return data
        
        # Korrelations-Matrix berechnen
        correlation_matrix = data[feature_cols].corr().abs()
        
        # Features zum Entfernen identifizieren
        features_to_remove = set()
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > threshold:
                    # Entferne Feature mit höherer durchschnittlicher Korrelation
                    feat1 = correlation_matrix.columns[i]
                    feat2 = correlation_matrix.columns[j]
                    
                    mean_corr1 = correlation_matrix[feat1].mean()
                    mean_corr2 = correlation_matrix[feat2].mean()
                    
                    if mean_corr1 > mean_corr2:
                        features_to_remove.add(feat1)
                    else:
                        features_to_remove.add(feat2)
        
        # Features entfernen
        remaining_features = ['Kunde'] + [f for f in feature_cols if f not in features_to_remove]
        filtered_data = data[remaining_features].copy()
        
        self.logger.info(f"✅ {len(features_to_remove)} korrelierte Features entfernt, {len(remaining_features)-1} verbleiben")
        return filtered_data
    
    def remove_low_variance_features(self, data: pd.DataFrame, 
                                   threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Entfernt Features mit niedriger Varianz
        
        Args:
            data: Feature-DataFrame
            threshold: Varianz-Schwellwert
            
        Returns:
            DataFrame ohne niedrig-variante Features
        """
        if threshold is None:
            threshold = self.feature_config['variance_threshold']
        
        self.logger.info(f"📊 Entferne Features mit niedriger Varianz (threshold={threshold})")
        
        feature_cols = [col for col in data.columns if col != 'Kunde']
        
        if len(feature_cols) == 0:
            return data
        
        # Varianz-Filter anwenden
        selector = VarianceThreshold(threshold=threshold)
        
        try:
            selected_features = selector.fit_transform(data[feature_cols])
            selected_mask = selector.get_support()
            
            remaining_features = ['Kunde'] + [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
            filtered_data = data[remaining_features].copy()
            
            removed_count = len(feature_cols) - sum(selected_mask)
            self.logger.info(f"✅ {removed_count} Features mit niedriger Varianz entfernt, {sum(selected_mask)} verbleiben")
            
            return filtered_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Varianz-Filter fehlgeschlagen: {e}")
            return data
    
    # =============================================================================
    # FEATURE PREPROCESSING
    # =============================================================================
    
    def standardize_features(self, data: pd.DataFrame, 
                           exclude_binary: bool = True) -> pd.DataFrame:
        """
        Standardisiert Features (intelligente Behandlung von One-Hot)
        
        Args:
            data: Feature-DataFrame
            exclude_binary: Schließt binäre Features aus (One-Hot)
            
        Returns:
            Standardisierte Features
        """
        self.logger.info("📏 Standardisiere Features")
        
        feature_cols = [col for col in data.columns if col != 'Kunde']
        standardized_data = data.copy()
        
        if len(feature_cols) == 0:
            return data
        
        # Binäre Features identifizieren (One-Hot)
        binary_features = []
        continuous_features = []
        
        for col in feature_cols:
            unique_values = data[col].dropna().unique()
            if exclude_binary and len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values):
                binary_features.append(col)
            else:
                continuous_features.append(col)
        
        # Nur kontinuierliche Features standardisieren
        if continuous_features:
            scaler = StandardScaler()
            standardized_values = scaler.fit_transform(data[continuous_features])
            standardized_data[continuous_features] = standardized_values
            
            self.scaler = scaler  # Für späteren Export
            
            self.logger.info(f"✅ {len(continuous_features)} kontinuierliche Features standardisiert")
            self.logger.info(f"   📋 {len(binary_features)} binäre Features unverändert")
        else:
            self.logger.info("ℹ️ Keine kontinuierlichen Features für Standardisierung gefunden")
        
        return standardized_data
    
    def handle_missing_values(self, data: pd.DataFrame, 
                            strategy: str = 'median') -> pd.DataFrame:
        """
        Behandelt fehlende Werte
        
        Args:
            data: Feature-DataFrame
            strategy: Strategie ('median', 'mean', 'forward_fill', 'zero')
            
        Returns:
            DataFrame ohne fehlende Werte
        """
        self.logger.info(f"🔧 Behandle fehlende Werte (Strategie: {strategy})")
        
        missing_before = data.isnull().sum().sum()
        
        if missing_before == 0:
            self.logger.info("ℹ️ Keine fehlenden Werte gefunden")
            return data
        
        feature_cols = [col for col in data.columns if col != 'Kunde']
        processed_data = data.copy()
        
        if strategy == 'median':
            processed_data[feature_cols] = processed_data[feature_cols].fillna(processed_data[feature_cols].median())
        elif strategy == 'mean':
            processed_data[feature_cols] = processed_data[feature_cols].fillna(processed_data[feature_cols].mean())
        elif strategy == 'forward_fill':
            processed_data[feature_cols] = processed_data[feature_cols].fillna(method='ffill')
        elif strategy == 'zero':
            processed_data[feature_cols] = processed_data[feature_cols].fillna(0)
        else:
            self.logger.warning(f"⚠️ Unbekannte Strategie {strategy}, verwende 'zero'")
            processed_data[feature_cols] = processed_data[feature_cols].fillna(0)
        
        missing_after = processed_data.isnull().sum().sum()
        
        self.logger.info(f"✅ {missing_before} fehlende Werte behandelt ({missing_after} verbleiben)")
        return processed_data
    
    def detect_outliers(self, data: pd.DataFrame, 
                       method: str = 'iqr') -> pd.DataFrame:
        """
        Erkennt und behandelt Ausreißer
        
        Args:
            data: Feature-DataFrame
            method: Methode ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            DataFrame mit behandelten Ausreißern
        """
        self.logger.info(f"🎯 Erkenne Ausreißer (Methode: {method})")
        
        feature_cols = [col for col in data.columns if col != 'Kunde']
        processed_data = data.copy()
        outliers_found = 0
        
        if method == 'iqr':
            for col in feature_cols:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)
                outliers_count = outliers.sum()
                
                if outliers_count > 0:
                    # Ausreißer durch Quantile ersetzen
                    processed_data.loc[processed_data[col] < lower_bound, col] = lower_bound
                    processed_data.loc[processed_data[col] > upper_bound, col] = upper_bound
                    outliers_found += outliers_count
        
        elif method == 'zscore':
            for col in feature_cols:
                z_scores = np.abs((processed_data[col] - processed_data[col].mean()) / processed_data[col].std())
                outliers = z_scores > 3
                outliers_count = outliers.sum()
                
                if outliers_count > 0:
                    # Ausreißer durch Median ersetzen
                    median_val = processed_data[col].median()
                    processed_data.loc[outliers, col] = median_val
                    outliers_found += outliers_count
        
        self.logger.info(f"✅ {outliers_found} Ausreißer behandelt")
        return processed_data
    
    # =============================================================================
    # MAIN PIPELINE
    # =============================================================================
    
    def create_cox_features(self, survival_panel: pd.DataFrame, 
                          stage0_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Hauptfunktion: Erstellt alle Cox-Features
        
        Args:
            survival_panel: Cox-Survival-Panel
            stage0_data: Stage0-Daten für Rolling-Features (optional)
            
        Returns:
            Vollständiger Feature-Datensatz für Cox-Modell mit Spalten:
            - Kunde: Kunden-ID
            - [FEATURE_NAMES]: Alle erstellten Features
        """
        self.logger.info("🚀 Starte Cox-Feature-Engineering-Pipeline")
        start_time = datetime.now()
        
        # Basis-DataFrame mit Kunden aus Survival-Panel
        base_customers = survival_panel[['Kunde']].drop_duplicates()
        self.logger.info(f"📊 Basis: {len(base_customers)} Kunden aus Survival-Panel")
        
        # 1. ROLLING-FEATURES erstellen (aus Stage0-Daten mit Loopback)
        if stage0_data is not None:
            self.logger.info("🔄 Schritt 1: Rolling-Features (ohne Loopback für unabhängige Features)")
            # Direkte Features statt Rolling (bessere Performance)
            rolling_features = self.create_direct_features(stage0_data, survival_panel)
            features_df = base_customers.merge(rolling_features, on='Kunde', how='left')
        else:
            self.logger.warning("⚠️ Keine Stage0-Daten für Rolling-Features")
            features_df = base_customers.copy()
        
        # 2. ONE-HOT FEATURES erstellen
        if stage0_data is not None:
            self.logger.info("🏷️ Schritt 2: One-Hot Features")
            onehot_features = self.create_onehot_features(stage0_data)
            features_df = features_df.merge(onehot_features, on='Kunde', how='left')
        
        # 3. ENHANCED FEATURES integrieren (falls verfügbar)
        if self.feature_config['use_enhanced_features']:
            self.logger.info("⚡ Schritt 3: Enhanced Features")
            try:
                enhanced_features = self.create_enhanced_features(stage0_data or survival_panel)
                if isinstance(enhanced_features, pd.DataFrame) and len(enhanced_features.columns) > 1:  # Mehr als nur 'Kunde'
                    features_df = features_df.merge(enhanced_features, on='Kunde', how='left')
                    self.logger.info(f"✅ Enhanced Features integriert: {len(enhanced_features.columns)-1} Features")
                else:
                    self.logger.warning("⚠️ Enhanced Features übersprungen: Keine gültigen Features")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced Features Integration fehlgeschlagen: {e}")
                # Fortfahren ohne Enhanced Features
        
        # 4. MISSING VALUES behandeln
        self.logger.info("🔧 Schritt 4: Missing Values")
        features_df = self.handle_missing_values(features_df, strategy='zero')
        
        # 5. OUTLIERS behandeln (temporär deaktiviert für Debugging)
        self.logger.info("🎯 Schritt 5: Ausreißer-Behandlung (übersprungen)")
        # features_df = self.detect_outliers(features_df, method='iqr')  # Temporär deaktiviert
        
        # 6. LOW VARIANCE Features entfernen
        if self.feature_config.get('remove_low_variance', True):
            self.logger.info("📊 Schritt 6: Niedrig-variante Features")
            features_df = self.remove_low_variance_features(features_df)
        
        # 7. KORRELIERTE Features entfernen
        if self.feature_config['remove_high_correlation']:
            self.logger.info("🔗 Schritt 7: Korrelierte Features")
            features_df = self.remove_correlated_features(features_df)
        
        # 8. FEATURE SELECTION
        if len([col for col in features_df.columns if col != 'Kunde']) > self.feature_config['feature_selection_k']:
            self.logger.info("🎯 Schritt 8: Feature-Selektion")
            
            # Dummy-Target für Feature-Selection (verwende duration aus survival_panel)
            target_data = survival_panel[['Kunde', 'duration']].drop_duplicates()
            merged_for_selection = features_df.merge(target_data, on='Kunde', how='inner')
            
            if len(merged_for_selection) > 0:
                selected_features = self.select_best_features(
                    merged_for_selection.drop('duration', axis=1), 
                    merged_for_selection['duration'],
                    method='combined'
                )
                self.selected_features = selected_features
                features_df = features_df[['Kunde'] + selected_features]
        
        # 9. STANDARDISIERUNG
        if self.feature_config['standardize_features']:
            self.logger.info("📏 Schritt 9: Standardisierung")
            features_df = self.standardize_features(features_df, exclude_binary=True)
        
        # 10. FINAL VALIDATION
        self.logger.info("✅ Schritt 10: Finale Validierung")
        validation_report = self.validate_features(features_df)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        feature_count = len([col for col in features_df.columns if col != 'Kunde'])
        self.logger.info(f"🎉 Cox-Feature-Engineering abgeschlossen!")
        self.logger.info(f"   📊 {feature_count} finale Features erstellt")
        self.logger.info(f"   👥 {len(features_df)} Kunden verarbeitet")
        self.logger.info(f"   ⏱️ Ausführungszeit: {execution_time:.2f}s")
        
        # Feature-Statistiken speichern
        self.feature_stats = {
            'total_features': feature_count,
            'customers_count': len(features_df),
            'execution_time': execution_time,
            'validation_report': validation_report,
            'selected_features': self.selected_features
        }
        
        return features_df
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Erstellt Feature-Importance-Report
        
        Returns:
            Report mit:
            - selected_features: Liste der finalen Features
            - feature_categories: Kategorisierung
            - importance_scores: Importance-Werte
            - preprocessing_stats: Preprocessing-Statistiken
        """
        feature_cols = [col for col in self.selected_features if col != 'Kunde']
        
        # Kategorisierung der finalen Features
        categories = self.categorize_enhanced_features(feature_cols)
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_features': len(feature_cols),
                'feature_selection_k': self.feature_config['feature_selection_k']
            },
            'selected_features': feature_cols,
            'feature_categories': categories,
            'importance_scores': self.feature_importance,
            'preprocessing_stats': self.feature_stats,
            'configuration': self.feature_config
        }
        
        return report
    
    # =============================================================================
    # UTILITIES & VALIDATION
    # =============================================================================
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert erstellte Features
        
        Args:
            features: Feature-DataFrame
            
        Returns:
            Validierungs-Report
        """
        feature_cols = [col for col in features.columns if col != 'Kunde']
        
        validation_report = {
            'total_features': len(feature_cols),
            'total_customers': len(features),
            'missing_values': features[feature_cols].isnull().sum().sum(),
            'infinite_values': np.isinf(features[feature_cols].select_dtypes(include=[np.number])).sum().sum(),
            'constant_features': sum((features[col].nunique() <= 1) for col in feature_cols),
            'feature_dtypes': features[feature_cols].dtypes.value_counts().to_dict(),
            'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Qualitäts-Score berechnen
        quality_score = 1.0
        if validation_report['missing_values'] > 0:
            quality_score -= 0.1
        if validation_report['infinite_values'] > 0:
            quality_score -= 0.2
        if validation_report['constant_features'] > 0:
            quality_score -= 0.1
        
        validation_report['quality_score'] = max(0.0, quality_score)
        
        return validation_report
    
    def export_feature_config(self, output_path: Optional[Path] = None) -> Path:
        """
        Exportiert Feature-Konfiguration
        
        Args:
            output_path: Ausgabe-Pfad (optional)
            
        Returns:
            Pfad zur exportierten Konfiguration
        """
        if output_path is None:
            if self.paths:
                output_dir = self.paths.dynamic_outputs_directory() / "cox_analysis"
            else:
                output_dir = Path("dynamic_system_outputs/cox_analysis")
            
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"cox_feature_config_{timestamp}.json"
        
        config_export = {
            'feature_config': self.feature_config,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'feature_stats': self.feature_stats,
            'data_dictionary_loaded': self.data_dictionary is not None,
            'enhanced_features_loaded': self.enhanced_features is not None
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_export, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📁 Feature-Konfiguration exportiert: {output_path}")
        return output_path


# =============================================================================
# FEATURE NAMING CONVENTIONS
# =============================================================================

class FeatureNamingConventions:
    """Standardisierte Feature-Namenskonventionen"""
    
    # Format: [BASE]_[OPERATION]_[WINDOW]_[AGGREGATION]
    ROLLING_FORMAT = "{base}_rolling_{window}p_{agg}"
    ACTIVITY_FORMAT = "{base}_activity_rate_{window}p"
    TREND_FORMAT = "{base}_trend_{window}p"
    ONEHOT_FORMAT = "{base}_{value}"
    ENHANCED_FORMAT = "{base}_{type}_{window}p"
    
    @staticmethod
    def standardize_name(name: str) -> str:
        """Standardisiert Feature-Namen"""
        # Entferne Sonderzeichen und normalisiere
        standardized = name.strip().replace(' ', '_').replace('-', '_')
        standardized = ''.join(char for char in standardized if char.isalnum() or char == '_')
        return standardized
    
    @staticmethod
    def parse_feature_name(name: str) -> Dict[str, str]:
        """Parst Feature-Namen in Komponenten"""
        parts = name.split('_')
        
        result = {
            'base': '',
            'operation': '',
            'window': '',
            'aggregation': ''
        }
        
        if len(parts) >= 1:
            result['base'] = parts[0]
        if len(parts) >= 2:
            result['operation'] = parts[1]
        if len(parts) >= 3 and parts[2].endswith('p'):
            result['window'] = parts[2]
        if len(parts) >= 4:
            result['aggregation'] = parts[3]
        
        return result


if __name__ == "__main__":
    # Test der Feature Engine
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Cox Feature Engine Test")
    
    # Feature Engine initialisieren
    engine = CoxFeatureEngine()
    
    # Test-Daten erstellen
    test_data = pd.DataFrame({
        'Kunde': [1, 1, 1, 2, 2, 2],
        'I_TIMEBASE': [202001, 202002, 202003, 202001, 202002, 202003],
        'I_MAINTENANCE': [1, 0, 1, 0, 1, 0],
        'N_DIGITALIZATIONRATE': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    })
    
    survival_panel = pd.DataFrame({
        'Kunde': [1, 2],
        'duration': [12, 8],
        'event': [1, 0]
    })
    
    # Feature-Engineering testen
    try:
        features = engine.create_cox_features(survival_panel, test_data)
        print(f"✅ Test erfolgreich: {len(features.columns)-1} Features erstellt")
        print(f"   Features: {[col for col in features.columns if col != 'Kunde']}")
        
        # Report erstellen
        report = engine.get_feature_importance_report()
        print(f"📊 Report erstellt: {report['metadata']['total_features']} Features")
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
