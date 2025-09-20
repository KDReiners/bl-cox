#!/usr/bin/env python3
"""
Cox Data Loader - Zentrales Datenmanagement für Cox-Analyse
============================================================

Konsolidiert Panel-Erstellung aus cox_panel_creator.py und stellt
einheitliche, validierte Datenstrukturen für Cox-Analyse bereit.

Kernfunktionen:
- Stage0-Daten laden und validieren
- Cox-Survival-Panel erstellen und cachen
- Datenqualität sicherstellen
- Einheitliche Zeitberechnungen (YYYYMM)

Autor: AI Assistant
Datum: 2025-01-27
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
import sys
import warnings

# Projekt-Pfade hinzufügen
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.paths_config import ProjectPaths
    from bl.Cox.cox_constants import (
        HUNDRED_PERCENT, FALSE_VALUE, TRUE_VALUE, START_INDEX
    )
except ImportError as e:
    print(f"⚠️ Import-Fehler: {e}")
    # Fallback-Konstanten
    HUNDRED_PERCENT = 100
    FALSE_VALUE = 0
    TRUE_VALUE = 1
    START_INDEX = 1


class CoxDataLoader:
    """
    Zentraler Datenmanager für Cox-Survival-Analyse
    """
    
    def __init__(self, cutoff_exclusive: int = 202501):
        """
        Initialisiert den Data Loader
        
        Args:
            cutoff_exclusive: Cutoff-Zeitpunkt (YYYYMM, exklusiv)
        """
        self.cutoff_exclusive = cutoff_exclusive
        self.logger = self._setup_logging()
        
        # Paths
        try:
            self.paths = ProjectPaths()
            self.stage0_cache_dir = self.paths.dynamic_outputs_directory() / "stage0_cache"
            self.cox_output_dir = self.paths.dynamic_outputs_directory() / "cox_survival_data"
            self.models_dir = self.paths.models_directory()
        except:
            self.stage0_cache_dir = Path("dynamic_system_outputs/stage0_cache")
            self.cox_output_dir = Path("dynamic_system_outputs/cox_survival_data")
            self.models_dir = Path("models")
        
        # Erstelle Output-Verzeichnisse
        self.cox_output_dir.mkdir(exist_ok=True)
        
        # State
        self.stage0_data: Optional[pd.DataFrame] = None
        self.survival_panel: Optional[pd.DataFrame] = None
        self.data_quality_report: Dict[str, Any] = {}
        self.cache_info: Dict[str, Any] = {}
        
        self.logger.info(f"📊 Cox Data Loader initialisiert (Cutoff: {cutoff_exclusive})")
    
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
    # STAGE0 DATA LOADING
    # =============================================================================
    
    def load_stage0_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Lädt Stage0-Rohdaten aus Cache
        
        Args:
            force_reload: Erzwingt Neuladen (ignoriert Cache)
            
        Returns:
            DataFrame mit Stage0-Daten
            
        Raises:
            FileNotFoundError: Wenn keine Stage0-Daten gefunden
            ValueError: Bei Datenvalidierungs-Fehlern
        """
        if not force_reload and self.stage0_data is not None:
            self.logger.info("📂 Verwende gecachte Stage0-Daten")
            return self.stage0_data
        
        self.logger.info("📂 Lade Stage0-Daten aus Cache")
        
        # Finde neueste Stage0-Cache-Datei
        if not self.stage0_cache_dir.exists():
            raise FileNotFoundError(f"Stage0-Cache-Verzeichnis nicht gefunden: {self.stage0_cache_dir}")
        
        cache_files = list(self.stage0_cache_dir.glob("*.json"))
        if not cache_files:
            raise FileNotFoundError(f"Keine Stage0-Cache-Dateien gefunden in: {self.stage0_cache_dir}")
        
        # Neueste Datei wählen (nach Änderungszeit)
        latest_file = max(cache_files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"   📁 Verwende: {latest_file.name}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Panel-Daten extrahieren - versuche verschiedene Strukturen
            panel_data = self._extract_panel_data(data)
            
            if not panel_data:
                raise ValueError("Keine Panel-Daten in Cache-Datei gefunden")
            
            # DataFrame erstellen
            self.stage0_data = pd.DataFrame(panel_data)
            
            # Validierung
            validation_report = self.validate_stage0_data(self.stage0_data)
            self.data_quality_report['stage0_validation'] = validation_report
            
            self.logger.info(f"✅ Stage0-Daten geladen: {len(self.stage0_data)} Records")
            self.logger.info(f"   👥 Kunden: {self.stage0_data['Kunde'].nunique()}")
            self.logger.info(f"   📅 Zeitraum: {self.stage0_data['I_TIMEBASE'].min()} - {self.stage0_data['I_TIMEBASE'].max()}")
            
            # Cache-Info speichern
            self.cache_info['stage0_file'] = str(latest_file)
            self.cache_info['stage0_loaded_at'] = datetime.now().isoformat()
            self.cache_info['stage0_records'] = len(self.stage0_data)
            
            return self.stage0_data
            
        except Exception as e:
            raise ValueError(f"Fehler beim Laden der Stage0-Daten: {e}")
    
    def _extract_panel_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Extrahiert Panel-Daten aus verschiedenen JSON-Strukturen
        
        Args:
            data: JSON-Daten
            
        Returns:
            Liste der Panel-Datensätze oder None
        """
        # Versuche verschiedene Schlüssel
        possible_keys = [
            'panel_data', 'data', 'customers_data', 'all_customers_data', 
            'complete_data', 'records', 'datensätze'
        ]
        
        for key in possible_keys:
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                if isinstance(data[key][0], dict) and 'Kunde' in data[key][0]:
                    self.logger.info(f"   ✅ Panel-Daten gefunden in: {key}")
                    return data[key]
        
        # Fallback: Suche nach beliebigen Listen mit Kundendaten
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and 'Kunde' in value[0]:
                    self.logger.info(f"   ✅ Panel-Daten gefunden in: {key} (Fallback)")
                    return value
        
        return None
    
    def validate_stage0_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert Stage0-Datenqualität
        
        Args:
            data: Stage0-DataFrame
            
        Returns:
            Validierungsreport mit Metriken und Warnungen
        """
        self.logger.info("🔍 Validiere Stage0-Datenqualität")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'unique_customers': data['Kunde'].nunique() if 'Kunde' in data.columns else 0,
            'required_columns_present': True,
            'missing_columns': [],
            'data_issues': [],
            'quality_score': 1.0
        }
        
        # Erforderliche Spalten prüfen
        required_columns = ['Kunde', 'I_TIMEBASE', 'I_Alive']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            report['required_columns_present'] = False
            report['missing_columns'] = missing_columns
            report['quality_score'] -= 0.5
            self.logger.error(f"❌ Fehlende erforderliche Spalten: {missing_columns}")
        
        if 'Kunde' in data.columns:
            # Duplikate prüfen
            duplicates = data.duplicated(subset=['Kunde', 'I_TIMEBASE']).sum()
            if duplicates > 0:
                report['data_issues'].append(f"Duplikate: {duplicates} Records")
                report['quality_score'] -= 0.1
                self.logger.warning(f"⚠️ {duplicates} duplizierte Records gefunden")
        
        if 'I_TIMEBASE' in data.columns:
            # Zeitbasis-Format prüfen
            invalid_timebase = data[~data['I_TIMEBASE'].astype(str).str.match(r'^\d{6}$')]
            if len(invalid_timebase) > 0:
                report['data_issues'].append(f"Ungültige I_TIMEBASE: {len(invalid_timebase)} Records")
                report['quality_score'] -= 0.2
                self.logger.warning(f"⚠️ {len(invalid_timebase)} Records mit ungültiger I_TIMEBASE")
        
        # Missing Values prüfen
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            report['data_issues'].append(f"Fehlende Werte: {missing_values}")
            report['quality_score'] -= min(0.2, missing_values / len(data))
            self.logger.warning(f"⚠️ {missing_values} fehlende Werte gefunden")
        
        # Qualitäts-Bewertung
        if report['quality_score'] >= 0.9:
            quality_level = "Excellent"
        elif report['quality_score'] >= 0.7:
            quality_level = "Good" 
        elif report['quality_score'] >= 0.5:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        report['quality_level'] = quality_level
        report['validation_passed'] = report['quality_score'] >= 0.5
        
        self.logger.info(f"✅ Datenvalidierung abgeschlossen: {quality_level} (Score: {report['quality_score']:.3f})")
        
        return report
    
    # =============================================================================
    # COX SURVIVAL PANEL CREATION  
    # =============================================================================
    
    def create_survival_panel_only_alive(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Erstellt Cox-Survival-Panel nur mit Kunden die IMMER I_Alive = True hatten
        (Echte Survival-Analyse ohne Data Leakage)
        """
        self.logger.info("🔄 Erstelle Cox-Survival-Panel (nur durchgehend lebende Kunden)")
        
        if data is None:
            data = self.load_stage0_data()
        
        if 'Kunde' not in data.columns or 'I_TIMEBASE' not in data.columns or 'I_Alive' not in data.columns:
            raise ValueError("Erforderliche Spalten für Panel-Erstellung fehlen: Kunde, I_TIMEBASE, I_Alive")
        
        survival_records = []
        customers = data['Kunde'].unique()
        
        # Bestimme globalen Zeitbereich
        global_start = data['I_TIMEBASE'].min()
        global_end = self.cutoff_exclusive
        
        self.logger.info(f"   👥 Prüfe {len(customers)} Kunden auf durchgehend I_Alive = True")
        self.logger.info(f"   📅 Zeitbereich: {global_start} bis {global_end} (exklusiv)")
        
        always_alive_count = 0
        
        for kunde in customers:
            kunde_data = data[data['Kunde'] == kunde].sort_values('I_TIMEBASE')
            
            if len(kunde_data) == 0:
                continue
            
            # Prüfe ob Kunde IMMER I_Alive = True hatte
            if not kunde_data['I_Alive'].all():
                continue  # Überspringe Kunden die jemals I_Alive = False hatten
            
            always_alive_count += 1
            
            # Survival-Record für durchgehend lebenden Kunden (alle zensiert)
            customer_start = kunde_data['I_TIMEBASE'].iloc[0]
            customer_end = kunde_data['I_TIMEBASE'].iloc[-1]
            
            # Prüfe ob Kunde vor Cutoff aktiv war
            if customer_start >= global_end:
                continue
            
            # Alle diese Kunden sind zensiert (event = 0)
            t_start = self.months_diff(global_start, customer_start)
            effective_end = min(customer_end, global_end - 1)
            t_end = self.months_diff(global_start, effective_end)
            duration = max(1, t_end - t_start)
            
            survival_record = {
                'Kunde': kunde,
                't_start': t_start,
                't_end': t_end,
                'duration': duration,
                'event': 0,  # Alle zensiert (kein Data Leakage)
                'I_Alive': True,
                'first_observed': customer_start,
                'last_observed': customer_end,
                'cutoff_used': global_end
            }
            
            survival_records.append(survival_record)
        
        if len(survival_records) == 0:
            raise ValueError("Keine durchgehend lebenden Kunden gefunden")
        
        # DataFrame erstellen
        self.survival_panel = pd.DataFrame(survival_records)
        
        # Validierung und Bereinigung
        self.survival_panel = self._validate_survival_panel(self.survival_panel)
        
        self.logger.info(f"✅ Cox-Survival-Panel erstellt: {len(self.survival_panel)} Records")
        self.logger.info(f"   👥 Durchgehend lebende Kunden: {always_alive_count}")
        self.logger.info(f"   📊 Events: {self.survival_panel['event'].sum()}")
        self.logger.info(f"   📊 Zensiert: {(self.survival_panel['event'] == 0).sum()}")
        self.logger.info(f"   🎯 KEIN DATA LEAKAGE: Alle I_Alive = True")
        
        return self.survival_panel

    def create_survival_panel(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Erstellt Cox-Survival-Panel aus Stage0-Daten
        
        Args:
            data: Stage0-Daten (optional, lädt automatisch wenn None)
            
        Returns:
            Cox-Survival-Panel mit Spalten:
            - Kunde: Kunden-ID
            - t_start: Start-Zeit (Monate seit Beginn)
            - t_end: Ende-Zeit (Monate seit Beginn)  
            - duration: Überlebensdauer in Monaten
            - event: Event-Indikator (1=Churn, 0=Zensiert)
            - I_Alive: Survival-Status
            
        Raises:
            ValueError: Bei Panel-Erstellungs-Fehlern
        """
        self.logger.info("🔄 Erstelle Cox-Survival-Panel")
        
        if data is None:
            data = self.load_stage0_data()
        
        if 'Kunde' not in data.columns or 'I_TIMEBASE' not in data.columns or 'I_Alive' not in data.columns:
            raise ValueError("Erforderliche Spalten für Panel-Erstellung fehlen: Kunde, I_TIMEBASE, I_Alive")
        
        survival_records = []
        customers = data['Kunde'].unique()
        
        self.logger.info(f"   👥 Verarbeite {len(customers)} Kunden")
        
        # Bestimme globalen Zeitbereich
        global_start = data['I_TIMEBASE'].min()
        global_end = self.cutoff_exclusive
        
        self.logger.info(f"   📅 Zeitbereich: {global_start} bis {global_end} (exklusiv)")
        
        for kunde in customers:
            kunde_data = data[data['Kunde'] == kunde].sort_values('I_TIMEBASE')
            
            if len(kunde_data) == 0:
                continue
            
            # Survival-Record für diesen Kunden erstellen
            survival_record = self._create_customer_survival_record(
                kunde=kunde,
                kunde_data=kunde_data,
                global_start=global_start,
                cutoff_exclusive=global_end
            )
            
            if survival_record:
                survival_records.append(survival_record)
        
        if not survival_records:
            raise ValueError("Keine Survival-Records erstellt - möglicherweise Datenprobleme")
        
        # DataFrame erstellen
        self.survival_panel = pd.DataFrame(survival_records)
        
        # Validierung und Bereinigung
        self.survival_panel = self._validate_survival_panel(self.survival_panel)
        
        self.logger.info(f"✅ Cox-Survival-Panel erstellt: {len(self.survival_panel)} Records")
        self.logger.info(f"   📊 Events: {self.survival_panel['event'].sum()}")
        self.logger.info(f"   📊 Zensiert: {(self.survival_panel['event'] == 0).sum()}")
        self.logger.info(f"   📊 Event-Rate: {self.survival_panel['event'].mean():.3f}")
        
        return self.survival_panel
    
    def _create_customer_survival_record(self, kunde: int, kunde_data: pd.DataFrame, 
                                       global_start: int, cutoff_exclusive: int) -> Optional[Dict[str, Any]]:
        """
        Erstellt Survival-Record für einen einzelnen Kunden
        
        Args:
            kunde: Kunden-ID
            kunde_data: Daten für diesen Kunden (sortiert nach I_TIMEBASE)
            global_start: Globaler Start-Zeitpunkt  
            cutoff_exclusive: Cutoff-Zeitpunkt (exklusiv)
            
        Returns:
            Survival-Record oder None bei Fehlern
        """
        try:
            # Bestimme ersten und letzten Zeitpunkt für Kunden
            customer_start = kunde_data['I_TIMEBASE'].iloc[0]
            customer_end = kunde_data['I_TIMEBASE'].iloc[-1]
            
            # Prüfe ob Kunde vor Cutoff aktiv war
            if customer_start >= cutoff_exclusive:
                return None  # Kunde startet nach Cutoff
            
            # Bestimme t_start (Monate seit global_start)
            t_start = self.months_diff(global_start, customer_start)
            
            # Bestimme Event und t_end
            last_alive_status = kunde_data['I_Alive'].iloc[-1]
            
            if last_alive_status == FALSE_VALUE:
                # Kunde ist gechurnt - Event-Zeit ist letzter beobachteter Zeitpunkt
                event = TRUE_VALUE
                t_end = self.months_diff(global_start, customer_end)
            else:
                # Kunde ist noch alive - zensiert zum Cutoff oder letzten Beobachtungszeitpunkt
                event = FALSE_VALUE
                # t_end ist der frühere von: Cutoff oder letzter Beobachtung
                effective_end = min(customer_end, cutoff_exclusive - 1)  # -1 weil cutoff exklusiv
                t_end = self.months_diff(global_start, effective_end)
            
            # Duration berechnen - robuste Berechnung
            raw_duration = t_end - t_start
            if raw_duration <= 0:
                # Fallback: Mindestens 1 Monat bei gleichen oder ungültigen Zeitpunkten
                duration = START_INDEX
                self.logger.debug(f"   ⚠️ Kunde {kunde}: Duration korrigiert ({raw_duration} → {duration})")
            else:
                duration = raw_duration
            
            # Survival-Record erstellen
            survival_record = {
                'Kunde': kunde,
                't_start': t_start,
                't_end': t_end,
                'duration': duration,
                'event': event,
                'I_Alive': last_alive_status,
                'first_observed': customer_start,
                'last_observed': customer_end,
                'cutoff_used': cutoff_exclusive
            }
            
            return survival_record
            
        except Exception as e:
            self.logger.warning(f"⚠️ Fehler bei Kunde {kunde}: {e}")
            return None
    
    def _validate_survival_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Validiert das erstellte Survival-Panel
        
        Args:
            panel: Survival-Panel DataFrame
            
        Returns:
            Bereinigtes Survival-Panel DataFrame
            
        Raises:
            ValueError: Bei Validierungsfehlern
        """
        # Erforderliche Spalten prüfen
        required_cols = ['Kunde', 't_start', 't_end', 'duration', 'event']
        missing_cols = [col for col in required_cols if col not in panel.columns]
        
        if missing_cols:
            raise ValueError(f"Fehlende Spalten im Survival-Panel: {missing_cols}")
        
        # Datentypen prüfen
        if not pd.api.types.is_integer_dtype(panel['duration']):
            raise ValueError("Duration muss Integer-Typ sein")
        
        if not panel['event'].isin([0, 1]).all():
            raise ValueError("Event muss 0 oder 1 sein")
        
        # Logische Konsistenz prüfen - Duration <= 0 Records herausfiltern
        invalid_duration_mask = (panel['duration'] <= 0)
        if invalid_duration_mask.any():
            invalid_count = invalid_duration_mask.sum()
            self.logger.warning(f"⚠️ {invalid_count} Records mit Duration <= 0 gefunden - werden entfernt")
            panel = panel[~invalid_duration_mask].copy()
            
            if len(panel) == 0:
                raise ValueError("Alle Records haben ungültige Duration <= 0")
        
        if (panel['t_end'] < panel['t_start']).any():
            invalid_count = (panel['t_end'] < panel['t_start']).sum()
            raise ValueError(f"{invalid_count} Records mit t_end < t_start gefunden")
        
        # Duplikate prüfen
        duplicates = panel['Kunde'].duplicated().sum()
        if duplicates > 0:
            raise ValueError(f"{duplicates} duplizierte Kunden im Panel gefunden")
        
        # Bereinigtes Panel zurückgeben
        return panel
    
    def load_existing_cox_panel(self, panel_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Lädt existierendes Cox-Panel aus JSON-Datei
        
        Args:
            panel_file: Pfad zur Panel-Datei (optional, sucht neueste)
            
        Returns:
            Cox-Survival-Panel DataFrame
            
        Raises:
            FileNotFoundError: Wenn kein Panel gefunden
        """
        self.logger.info("📂 Lade existierendes Cox-Panel")
        
        if panel_file is None:
            # Suche neueste Panel-Datei
            panel_files = list(self.cox_output_dir.glob("cox_survival_panel_*.json"))
            if not panel_files:
                raise FileNotFoundError(f"Keine Cox-Panel-Dateien gefunden in: {self.cox_output_dir}")
            
            panel_file = max(panel_files, key=lambda x: x.stat().st_mtime)
        
        self.logger.info(f"   📁 Verwende: {panel_file.name}")
        
        try:
            with open(panel_file, 'r', encoding='utf-8') as f:
                panel_data = json.load(f)
            
            # Panel-Daten extrahieren
            if 'survival_panel' in panel_data:
                records = panel_data['survival_panel']
            elif 'records' in panel_data:
                records = panel_data['records']
            else:
                records = panel_data  # Assume top-level is the data
            
            self.survival_panel = pd.DataFrame(records)
            
            # Validierung und Bereinigung
            self.survival_panel = self._validate_survival_panel(self.survival_panel)
            
            self.logger.info(f"✅ Cox-Panel geladen: {len(self.survival_panel)} Records")
            
            # Cache-Info aktualisieren
            self.cache_info['panel_file'] = str(panel_file)
            self.cache_info['panel_loaded_at'] = datetime.now().isoformat()
            
            return self.survival_panel
            
        except Exception as e:
            raise ValueError(f"Fehler beim Laden des Cox-Panels: {e}")
    
    def save_survival_panel(self, panel: pd.DataFrame, filename: Optional[str] = None) -> Path:
        """
        Speichert Cox-Survival-Panel als JSON
        
        Args:
            panel: Cox-Panel DataFrame
            filename: Dateiname (optional, auto-generiert mit Timestamp)
            
        Returns:
            Pfad zur gespeicherten Datei
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cox_survival_panel_v5_{timestamp}.json"
        
        output_path = self.cox_output_dir / filename
        
        # Panel-Daten mit Metadaten speichern
        panel_export = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'cutoff_exclusive': self.cutoff_exclusive,
                'total_records': len(panel),
                'unique_customers': panel['Kunde'].nunique(),
                'event_count': panel['event'].sum(),
                'event_rate': panel['event'].mean(),
                'version': 'v5'
            },
            'survival_panel': panel.to_dict('records'),
            'data_quality_report': self.data_quality_report,
            'cache_info': self.cache_info
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(panel_export, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Cox-Panel gespeichert: {output_path}")
        return output_path
    
    # =============================================================================
    # DATA QUALITY & UTILITIES
    # =============================================================================
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Erstellt Zusammenfassung der geladenen Daten
        
        Returns:
            Summary mit Kennzahlen
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'cutoff_exclusive': self.cutoff_exclusive
        }
        
        # Stage0-Daten Summary
        if self.stage0_data is not None:
            summary['stage0_data'] = {
                'total_records': len(self.stage0_data),
                'unique_customers': self.stage0_data['Kunde'].nunique(),
                'time_range': {
                    'start': int(self.stage0_data['I_TIMEBASE'].min()),
                    'end': int(self.stage0_data['I_TIMEBASE'].max())
                },
                'columns': list(self.stage0_data.columns),
                'memory_usage_mb': self.stage0_data.memory_usage(deep=True).sum() / 1024 / 1024
            }
        
        # Survival-Panel Summary
        if self.survival_panel is not None:
            summary['survival_panel'] = {
                'total_records': len(self.survival_panel),
                'unique_customers': self.survival_panel['Kunde'].nunique(),
                'event_count': int(self.survival_panel['event'].sum()),
                'event_rate': float(self.survival_panel['event'].mean()),
                'duration_stats': {
                    'min': int(self.survival_panel['duration'].min()),
                    'max': int(self.survival_panel['duration'].max()),
                    'mean': float(self.survival_panel['duration'].mean()),
                    'median': float(self.survival_panel['duration'].median())
                }
            }
        
        # Datenqualität
        summary['data_quality_score'] = self.data_quality_report.get('stage0_validation', {}).get('quality_score', 0.0)
        summary['cache_info'] = self.cache_info
        
        return summary
    
    def months_diff(self, tb_start: int, tb_end: int) -> int:
        """
        Berechnet Monats-Differenz zwischen YYYYMM-Zeitpunkten
        
        Args:
            tb_start: Start-Zeitpunkt (YYYYMM)
            tb_end: Ende-Zeitpunkt (YYYYMM)
            
        Returns:
            Anzahl Monate zwischen den Zeitpunkten
        """
        start_year = tb_start // HUNDRED_PERCENT
        start_month = tb_start % HUNDRED_PERCENT
        end_year = tb_end // HUNDRED_PERCENT
        end_month = tb_end % HUNDRED_PERCENT
        
        return max(0, (end_year - start_year) * 12 + (end_month - start_month))
    
    def get_alive_customers_at_cutoff(self) -> List[int]:
        """
        Ermittelt alle lebenden Kunden zum Cutoff-Zeitpunkt
        
        Returns:
            Liste der Kunden-IDs, die am Cutoff noch aktiv sind
        """
        if self.stage0_data is None:
            self.load_stage0_data()
        
        # Kunden, die am letzten Zeitpunkt vor Cutoff noch I_Alive=True haben
        cutoff_data = self.stage0_data[self.stage0_data['I_TIMEBASE'] < self.cutoff_exclusive]
        
        if len(cutoff_data) == 0:
            return []
        
        # Letzten Status pro Kunde vor Cutoff ermitteln
        last_status = cutoff_data.groupby('Kunde').agg({
            'I_TIMEBASE': 'max',
            'I_Alive': 'last'
        }).reset_index()
        
        # Nur Kunden mit I_Alive=True
        alive_customers = last_status[last_status['I_Alive'] == TRUE_VALUE]['Kunde'].tolist()
        
        self.logger.info(f"👥 {len(alive_customers)} lebende Kunden zum Cutoff {self.cutoff_exclusive}")
        return alive_customers
    
    # =============================================================================
    # CACHING & PERFORMANCE
    # =============================================================================
    
    def clear_cache(self):
        """Löscht alle gecachten Daten"""
        self.stage0_data = None
        self.survival_panel = None
        self.data_quality_report = {}
        self.cache_info = {}
        self.logger.info("🗑️ Cache geleert")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Informationen über gecachte Daten
        
        Returns:
            Cache-Info mit Timestamps und Größen
        """
        info = self.cache_info.copy()
        
        # Aktuelle Memory-Usage hinzufügen
        if self.stage0_data is not None:
            info['stage0_memory_mb'] = self.stage0_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        if self.survival_panel is not None:
            info['panel_memory_mb'] = self.survival_panel.memory_usage(deep=True).sum() / 1024 / 1024
        
        return info
    
    def preload_all_data(self) -> Dict[str, Any]:
        """
        Lädt alle Daten vor und erstellt Summary
        
        Returns:
            Vollständiger Data-Load-Report
        """
        self.logger.info("🚀 Starte vollständiges Daten-Preloading")
        start_time = datetime.now()
        
        try:
            # 1. Stage0-Daten laden
            self.load_stage0_data()
            
            # 2. Survival-Panel erstellen
            self.create_survival_panel()
            
            # 3. Summary erstellen
            summary = self.get_data_summary()
            
            end_time = datetime.now()
            load_time = (end_time - start_time).total_seconds()
            
            report = {
                'success': True,
                'load_time_seconds': load_time,
                'data_summary': summary,
                'message': f"Alle Daten erfolgreich geladen in {load_time:.2f}s"
            }
            
            self.logger.info(f"✅ Preloading abgeschlossen in {load_time:.2f}s")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Preloading fehlgeschlagen: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Preloading fehlgeschlagen: {e}"
            }


# =============================================================================
# DATENSTRUKTUREN & SCHEMAS
# =============================================================================

class CoxPanelSchema:
    """Schema-Definition für Cox-Survival-Panel"""
    
    REQUIRED_COLUMNS = [
        'Kunde',         # int: Kunden-ID
        't_start',       # int: Start-Zeit in Monaten
        't_end',         # int: Ende-Zeit in Monaten
        'duration',      # int: Überlebensdauer in Monaten
        'event',         # int: Event-Indikator (0/1)
        'I_Alive'        # bool: Survival-Status
    ]
    
    OPTIONAL_COLUMNS = [
        'first_observed',    # int: Erster beobachteter Zeitpunkt (YYYYMM)
        'last_observed',     # int: Letzter beobachteter Zeitpunkt (YYYYMM)
        'cutoff_used'        # int: Verwendeter Cutoff (YYYYMM)
    ]
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Validiert DataFrame gegen Schema"""
        missing_required = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in cls.REQUIRED_COLUMNS + cls.OPTIONAL_COLUMNS]
        
        return {
            'valid': len(missing_required) == 0,
            'missing_required': missing_required,
            'extra_columns': extra_columns,
            'total_columns': len(df.columns)
        }


class DataQualityMetrics:
    """Metriken für Datenqualitätsbewertung"""
    
    def __init__(self):
        self.missing_values_ratio: float = 0.0
        self.duplicate_customers_count: int = 0
        self.invalid_timebase_count: int = 0
        self.inconsistent_survival_count: int = 0
        self.overall_quality_score: float = 1.0
    
    @classmethod
    def calculate_metrics(cls, data: pd.DataFrame) -> 'DataQualityMetrics':
        """Berechnet Qualitäts-Metriken für DataFrame"""
        metrics = cls()
        
        if len(data) == 0:
            metrics.overall_quality_score = 0.0
            return metrics
        
        # Missing Values
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        metrics.missing_values_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        # Duplikate (wenn Kunde-Spalte vorhanden)
        if 'Kunde' in data.columns:
            metrics.duplicate_customers_count = data['Kunde'].duplicated().sum()
        
        # Ungültige Timebase (wenn I_TIMEBASE-Spalte vorhanden)
        if 'I_TIMEBASE' in data.columns:
            invalid_tb = data[~data['I_TIMEBASE'].astype(str).str.match(r'^\d{6}$')]
            metrics.invalid_timebase_count = len(invalid_tb)
        
        # Overall Quality Score berechnen
        score = 1.0
        score -= min(0.3, metrics.missing_values_ratio)  # Max 30% Abzug für Missing Values
        score -= min(0.2, metrics.duplicate_customers_count / len(data))  # Max 20% für Duplikate
        score -= min(0.2, metrics.invalid_timebase_count / len(data))  # Max 20% für ungültige Timebase
        
        metrics.overall_quality_score = max(0.0, score)
        
        return metrics


if __name__ == "__main__":
    # Test des Data Loaders
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Cox Data Loader Test")
    
    # Data Loader initialisieren
    loader = CoxDataLoader(cutoff_exclusive=202501)
    
    try:
        # Test: Daten laden
        report = loader.preload_all_data()
        
        if report['success']:
            print(f"✅ Test erfolgreich: {report['message']}")
            
            # Summary anzeigen
            summary = report['data_summary']
            if 'stage0_data' in summary:
                print(f"   📊 Stage0: {summary['stage0_data']['total_records']} Records")
            if 'survival_panel' in summary:
                print(f"   📊 Panel: {summary['survival_panel']['total_records']} Records")
                print(f"   📊 Event-Rate: {summary['survival_panel']['event_rate']:.3f}")
        else:
            print(f"❌ Test fehlgeschlagen: {report['message']}")
    
    except Exception as e:
        print(f"❌ Test-Fehler: {e}")
        import traceback
        traceback.print_exc()
