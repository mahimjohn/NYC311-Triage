# NYC 311 Service Request Triage and Resolution-Time Prediction System

## Overview

A production-oriented machine learning and decision-support system
for analyzing NYC 311 service requests and identifying requests
that are likely to take more than seven days to resolve.

## Problem

Municipal operations teams receive a large volume of 311 service
requests. The system helps analysts identify requests with a higher
likelihood of delayed resolution so that they can prioritize
operational attention.

## Core Prediction

Predict whether a newly created service request will take more than
7 days to resolve.

## Primary User

Municipal Operations Analyst.

## Technology Stack

- Python
- Pandas / NumPy
- Scikit-learn
- PostgreSQL
- FastAPI
- Streamlit
- Plotly / Folium
- Power BI / Tableau
- Git / GitHub

## Architecture

NYC Open Data
→ Data Ingestion
→ Data Processing
→ PostgreSQL
→ ML Prediction Service
→ FastAPI
→ Streamlit

## Status

🚧 Under Development