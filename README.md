# bl-cox - Cox-Survival-Analyse

**Last reviewed: 2025-09-29**

## 🎯 **Zweck**

Business-Logic für Cox-Survival-Analyse mit Customer Risk Profiling und Prioritization.

## 🏗️ **Architektur**

- **Survival-Analyse**: Cox-Proportional-Hazards-Model
- **Risk-Profiling**: Customer-spezifische Survival-Wahrscheinlichkeiten
- **Prioritization**: ROI-basierte Kunden-Priorisierung
- **Segmentierung**: Digitalization-basierte Cluster-Analyse

## 🚀 **Quick Start**

### **Pipeline starten:**
```bash
# Über UI
http://localhost:8080/ → Experiment auswählen → "Cox" starten

# Über API
curl -X POST http://localhost:5050/run/cox -d '{"experiment_id":1, "cutoff_exclusive":"202501"}'
```

### **Ergebnisse ansehen:**
- **Management Studio**: http://localhost:5051/sql/
- **Tabellen**: `cox_survival`, `cox_prioritization_results`

## 📊 **Output-Tabellen**

- `cox_survival`: Survival-Wahrscheinlichkeiten (6/12/18/24 Monate)
- `cox_prioritization_results`: ROI-basierte Kunden-Priorisierung
- `customer_cox_details_{experiment_id}`: Customer Risk Profiles
- `churn_cox_fusion`: Fusion-View (Churn + Cox)

## 🔧 **Konfiguration**

- **Survival-Horizonte**: 6, 12, 18, 24 Monate
- **Prioritization**: ROI-basierte Scoring
- **Segmentierung**: Digitalization-Cluster

## 📚 **Dokumentation**

**Zentrale Dokumentation:** [NEXT_STEPS.md](../NEXT_STEPS.md)

**Detaillierte Anleitungen:**
- [bl-cox/RUNBOOK.md](RUNBOOK.md) - Betriebsabläufe