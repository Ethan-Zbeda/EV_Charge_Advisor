# EV Charge Advisor

## Overview

EV Charge Advisor is an AI-powered forecasting and recommendation platform designed to improve the workplace EV charging experience. Instead of relying on incomplete real-time charger telemetry, the platform leverages historical charging data and machine learning to forecast charging demand, identify congestion patterns, and recommend optimal charging opportunities for employees.

The solution helps employees plan charging sessions more effectively while providing leadership with actionable insights into charger utilization, congestion trends, and future infrastructure needs.

---

## Problem Statement

Workplace EV charging presents several challenges:

- Limited real-time visibility into charger availability
- Growing EV adoption and increasing demand for charging resources
- Employees spending time searching for available chargers
- Limited data availability due to partially connected charging infrastructure

With hundreds of employees enrolled in the workplace charging program and limited charger telemetry, a forecasting-based approach provides a scalable solution for improving charger accessibility.

---

## Solution

EV Charge Advisor uses historical charging session data to:

- Forecast charging demand
- Predict congestion levels
- Recommend optimal charging windows
- Estimate charger availability probability
- Provide AI-generated recommendations and explanations
- Support infrastructure planning and future charger deployment decisions

---

# Employee Dashboard

The Employee Dashboard provides personalized charging guidance to help employees maximize their chances of successfully finding charging opportunities.

### Features

#### Office Selector
Allows employees to select their office location.

#### Day / Time Selector
Enables employees to choose a desired charging day and arrival time.

#### Forecasted Congestion
Displays predicted charging demand and congestion levels throughout the day.

#### Recommended Charging Window
Identifies the optimal charging window based on forecasted demand patterns.

#### Availability Probability
Provides an estimated probability of finding an available charging opportunity during the selected time period.

#### AI Explanation
Generates a natural language explanation describing:

- Why a recommendation was made
- Expected congestion trends
- Best charging opportunities
- Confidence in the recommendation

---

# Leadership Dashboard

The Leadership Dashboard provides strategic insights into workplace charging utilization and infrastructure planning.

### Features

Fill in as needed

---

# Model Validation

The forecasting engine includes validation metrics to ensure prediction quality and transparency.

### Baseline vs Model Performance

Compare:

- Historical average demand forecast
- Machine learning forecast

fill in as needed

---

# Business Value

## Time Saved

Reduces time spent searching for available chargers by helping employees identify optimal charging opportunities before arriving at a charging location.

## Reduced Employee Frustration

Provides greater visibility into charging demand and improves planning capabilities.

## Improved Charger Utilization

Encourages more balanced charger usage throughout the day by identifying lower-demand charging windows.

## Infrastructure Planning

Provides data-driven insights to support:

- Future charger expansion
- Site prioritization
- Capacity planning
- EV adoption forecasting

---

# Technical Approach

### Data Sources

- ChargePoint Charging Session Data
- Blink Charging Session Data
- EV Charger Work Orders
- Historical Charging Utilization Data

### Forecasting Inputs

- Office Location
- Day of Week
- Time of Day
- Historical Charging Sessions
- Seasonal Trends
- Charger Utilization Metrics

### Technologies

- Python
- Pandas
- Streamlit
- Plotly
- Scikit-Learn / XGBoost
- Azure OpenAI

---

# Future Enhancements

- Multi-office forecasting
- Real-time charger integrations
- Mobile application support
- Teams chatbot integration
- Personalized charging recommendations
- Infrastructure demand forecasting
- Reservation and scheduling capabilities

---

# Expected Outcomes

- Improved employee charging experience
- Reduced congestion during peak periods
- Better utilization of existing charging infrastructure
- Increased visibility into charging demand trends
- Data-driven infrastructure planning decisions

---

## Project Vision

> EV Charge Advisor empowers employees and leadership with AI-driven charging intelligence, transforming workplace EV charging from a reactive search process into a proactive planning experience.
