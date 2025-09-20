# Cox Survival Analysis Pipeline - Business Logic Module

```yaml
module_info:
  name: "Cox Survival Analysis System"
  purpose: "Time-based churn prediction with multi-horizon survival analysis"
  status: "PRODUCTION"
  integration_level: "CORE_COMPONENT"
  performance_target: "C-Index > 0.80"
  last_updated: "2025-09-18"
  ai_agent_optimized: true
```

## 🎯 **MODULE OVERVIEW**

### **Primary Functions:**
- **Cox Proportional Hazards Modeling** - Time-to-churn survival analysis
- **Multi-Horizon Predictions** - 6, 12, 18, 24 month survival probabilities
- **Risk Categorization** - Customer risk profiling (HIGH/MEDIUM/LOW)
- **Silent Churn Detection** - Identifies churns missed by binary classification
- **JSON-Database Integration** - Enterprise SQL analytics with DuckDB views

### **Business Impact:**
- **Time-Based Planning** - Intervention timing optimization
- **Resource Allocation** - Risk-based customer segmentation  
- **Silent Churn Intelligence** - Captures churns invisible to binary models
- **Customer Lifetime Prediction** - Expected survival duration calculation
- **OLAP Analytics** - Multi-dimensional business intelligence

## 🏗️ **ARCHITECTURE COMPONENTS**

### **Core Classes:**
```python
# Primary Pipeline Components
CoxAutoProcessor()           # Orchestrates experiment processing and status management
CoxDataLoader()             # Survival panel data construction
CoxFeatureEngine()          # Optimal 4-feature selection and engineering
CoxModelTrainer()           # Cox regression training and evaluation
CoxEvaluator()              # C-Index calculation and risk scoring

# Analysis Components
CoxMain()                   # Legacy comprehensive pipeline
CoxWorkingMain()            # Production-ready pipeline orchestrator
```

### **Data Flow:**
```yaml
input:
  - "Stage0 cache data (dynamic_system_outputs/stage0_cache/)"
  - "Experiment configuration from experiments table"
  - "Customer timebase data with I_Alive indicators"
  
process:
  1. "Survival panel construction (duration, event)"
  2. "Optimal 4-feature selection and engineering"
  3. "Cox proportional hazards model training"
  4. "Multi-horizon survival probability calculation"
  5. "Risk categorization and customer profiling"
  6. "SQL views generation for OLAP analytics"
  
output:
  - "JSON-Database tables (5 cox-specific tables)"
  - "Customer risk profiles and survival probabilities"
  - "DataCube SQL views for business intelligence"
```

## 🚀 **QUICK START FOR AI-AGENTS**

### **Basic Usage:**
```bash
# Environment setup
source churn_prediction_env/bin/activate
cd /Users/klaus.reiners/Projekte/Cursor\ ChurnPrediction\ -\ Reengineering

# Process single experiment
python bl/Cox/cox_working_main.py --experiment-id 8 --cutoff-months 24

# Auto-process all pending experiments
python bl/Cox/cox_auto_processor.py

# Check processing status
python bl/Cox/cox_auto_processor.py --status

# Validate cutoff parameters  
python bl/Cox/cox_auto_processor.py --check-cutoff
```

### **Programmatic API:**
```python
from bl.Cox.cox_auto_processor import CoxAutoProcessor
from bl.json_database.sql_query_interface import SQLQueryInterface

# Process experiments
processor = CoxAutoProcessor()
results = processor.run_all_pending_experiments()

# Query survival analysis
qi = SQLQueryInterface()
survival_data = qi.execute_query("SELECT * FROM cox_survival_enhanced WHERE id_experiments = 8")
risk_profile = qi.execute_query("SELECT * FROM customer_risk_profile WHERE id_experiments = 8")
```

## 📊 **CONFIGURATION & CONSTANTS**

### **Key Configuration Files:**
```yaml
config_files:
  constants: "bl/Cox/cox_constants.py"
  data_dictionary: "config/data_dictionary_optimized.json"
  paths: "config/paths_config.py"
  
optimal_features:
  - "I_SOCIALINSURANCENOTES_sum_1yr_before_timebase"
  - "I_UHD_sum_1yr_before_timebase" 
  - "I_UHD_sum_6m_before_timebase"
  - "N_DIGITALIZATIONRATE"
```

### **Performance Targets:**
```yaml
target_metrics:
  c_index: 0.80
  processing_time: "< 10 seconds per experiment"
  customer_coverage: "> 6000 customers"
  
survival_horizons:
  - 6   # months
  - 12  # months  
  - 18  # months
  - 24  # months
  
risk_thresholds:
  high_risk: "> 70% churn probability at 12 months"
  medium_risk: "40-70% churn probability at 12 months"
  low_risk: "< 40% churn probability at 12 months"
```

## 🔗 **SYSTEM INTEGRATION**

### **Database Schema:**
```yaml
json_database_tables:
  cox_survival: "Customer-level survival data (duration, event, probabilities)"
  cox_prioritization_results: "Risk scores and priority rankings"
  cox_analysis_metrics: "Model performance and feature importance"
  customer_cox_details: "Customer details with Cox predictions"
  experiment_kpis: "Experiment-level performance metrics"
  
sql_views:
  cox_survival_enhanced: "Multi-horizon survival probabilities"
  customer_risk_profile: "Risk categorization and business actions"
  cox_performance_summary: "Experiment performance aggregation"
  
foreign_keys:
  - "All tables: id_experiments → experiments.experiment_id"
  
status_management:
  - "experiments.status: 'created' → 'processing' → 'processed'"
```

### **Dependencies:**
```yaml
internal_dependencies:
  - "bl/json_database/churn_json_database.py"
  - "bl/json_database/sql_query_interface.py" 
  - "config/paths_config.py"
  
external_dependencies:
  - "scikit-survival >= 0.17"
  - "pandas >= 1.3"
  - "numpy >= 1.21"
  - "scipy >= 1.7"
```

## 📈 **PERFORMANCE & MONITORING**

### **Current Performance (Production):**
```yaml
model_performance:
  c_index: 0.8094
  consistency: "±0.0001 across runs"
  customer_coverage: 6287
  optimal_features: 4
  
system_performance:
  avg_processing_time: "6.5 seconds"
  memory_usage: "~45MB"
  success_rate: "100%"
  
business_integration:
  cox_churn_overlap: "74.8% (4,742 customers)"
  cox_only_silent_churns: "25.2% (1,598 customers)"
  total_analyzed_customers: 6340
```

### **DataCube Analytics Performance:**
```yaml
sql_performance:
  query_execution: "< 2 seconds complex queries"
  view_generation: "< 1 second"
  concurrent_users: "Multi-user ready"
  
auto_scaling:
  new_experiments: "Zero-touch integration"
  filter_mechanism: "model_type='cox_survival' AND backtest_from >= 202501"
  datacube_queries: "4 production-ready queries"
```

## 🔧 **TROUBLESHOOTING FOR AI-AGENTS**

### **Common Issues:**
```yaml
survival_panel_errors:
  symptom: "Duration calculation fails"
  solution: "Verify I_Alive column exists and timebase format is YYYYMM"
  
cox_model_errors:
  symptom: "Cox regression fails to converge" 
  solution: "Check feature scaling and remove constant features"
  
database_integration_errors:
  symptom: "JSON-DB write failures for Cox tables"
  solution: "Verify churn_json_database.py has Cox table schemas defined"
  
sql_view_errors:
  symptom: "DuckDB views not accessible"
  solution: "Check SQLQueryInterface connection and table existence"
```

### **Performance Optimization:**
```yaml
optimization_tips:
  - "Feature selection: Use only optimal 4 features for best performance"
  - "Memory management: Process customers in batches for large experiments"  
  - "SQL optimization: Use indexed queries on id_experiments"
  - "Parallel processing: Enable concurrent experiment processing"
```

## 📊 **DATACUBE ANALYTICS INTEGRATION**

### **Available SQL Views:**
```yaml
production_views:
  cox_survival_enhanced:
    description: "Multi-horizon survival probabilities"
    key_fields: ["Kunde", "P_Event_6m", "P_Event_12m", "survival_6m", "survival_12m"]
    
  customer_risk_profile:
    description: "Risk categorization with business actions"  
    key_fields: ["Kunde", "risk_category", "survival_12m_percent", "business_action"]
    
  cox_performance_summary:
    description: "Experiment performance aggregation"
    key_fields: ["id_experiments", "total_customers", "avg_survival_12m", "high_risk_count"]
```

### **Business Intelligence Queries:**
```sql
-- Daily monitoring dashboard
SELECT experiment_id, high_risk_count, avg_survival_percentage, alert_level
FROM cox_performance_summary_with_alerts 
WHERE created_date >= DATE('now', '-7 days');

-- Silent churn detection
SELECT churn_timeline, predictability_level, anzahl_silent_churns, business_action
FROM silent_churn_detection_production
WHERE predictability_level = 'UNEXPECTED';

-- Cox-Churn integration analysis  
SELECT customer_segment, anzahl_kunden, percentage
FROM cox_churn_integration_validation
ORDER BY anzahl_kunden DESC;
```

## 📋 **AI-AGENT MAINTENANCE CHECKLIST**

### **After Code Changes:**
```yaml
validation_steps:
  - "Run: python bl/Cox/cox_auto_processor.py --validate"
  - "Check: C-Index within 0.8094 ± 0.001 range"
  - "Verify: All 5 JSON-Database tables populated"
  - "Test: SQL views return expected data structure"
  - "Validate: Silent churn detection produces results"
  
update_requirements:
  - "Performance changes → Update C-Index metrics"
  - "New features → Update optimal_features list"
  - "SQL changes → Update views documentation"
  - "DataCube changes → Update analytics queries"
```

### **Auto-Scaling Verification:**
```yaml
new_experiment_check:
  - "Verify: New experiments auto-detected by filter"
  - "Check: DataCube queries include new experiment_id"
  - "Validate: No manual configuration required"
  - "Test: All 4 DataCube queries return updated results"
```

---

**📅 Last Updated:** 2025-09-18  
**🤖 Optimized for:** AI-Agent maintenance and usage  
**🎯 Status:** Production-ready core component  
**🔗 Related:** docs/COX_ARCHITECTURE_SPECIFICATION.md, docs/COX_SQL_INTEGRATION_COMPLETE.md
