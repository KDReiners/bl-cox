#!/usr/bin/env python3
"""
Cox Module Constants
===================

Zentrale Konstanten für alle Cox-Module.

Autor: AI Assistant
Datum: 2025-01-27
"""

# Boolean Values
FALSE_VALUE = 0
TRUE_VALUE = 1

# Index Values
START_INDEX = 0
FIRST_INDEX = 1
TWO_VALUE = 2

# Time Horizons (Months)
SIX_MONTH_HORIZON = 6
TWELVE_MONTH_HORIZON = 12
EIGHTEEN_MONTH_HORIZON = 18

# Percentage Thresholds
FIFTY_PERCENT_THRESHOLD = 50
SIXTY_PERCENT_THRESHOLD = 60
NINETY_NINE_PERCENT = 99
HUNDRED_PERCENT = 100

# Correlation/Variance Thresholds
HIGH_CORRELATION_THRESHOLD = 0.95
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-9

# Database Configuration
DEFAULT_SQL_PORT = 1433

# Feature Engineering
DEFAULT_FEATURE_WINDOW = 6
DEFAULT_TREND_WINDOW = 12
DEFAULT_ACTIVITY_WINDOW = 18

# Model Configuration
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SIGNIFICANCE_LEVEL = 0.05

# File Extensions
JSON_EXTENSION = '.json'
CSV_EXTENSION = '.csv'
PNG_EXTENSION = '.png'

# Data Types
FLOAT64_TYPE = 'float64'
INT64_TYPE = 'int64'

# Merge Operations
LEFT_MERGE = 'left'
RIGHT_MERGE = 'right'
INNER_MERGE = 'inner'

# Error Handling
DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_COUNT = 3
