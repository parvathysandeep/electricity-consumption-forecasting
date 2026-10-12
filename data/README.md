# Dataset Documentation

## Primary Dataset: Open Power System Data (OPSD) - Germany Time Series

- **Official Source**: [Open Power System Data Platform](https://data.open-power-system-data.org/time_series/)
- **Repository / Direct URL**: `https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv`
- **File Name**: `electricity_consumption.csv` (or `opsd_germany_daily.csv`)
- **Temporal Interval**: Daily observations (2006-01-01 to 2017-12-31)
- **Total Records**: 4,383 rows (12 full calendar years)
- **Target Variable**: `Consumption` (Electricity consumption in Gigawatt-hours, GWh)
- **Exogenous Columns**:
  - `Date`: ISO 8601 Date string (YYYY-MM-DD)
  - `Wind`: Wind power generation (GWh)
  - `Solar`: Solar power generation (GWh)
  - `Wind+Solar`: Total renewable power generation (GWh)

## Data Integrity & Preprocessing Notes
1. **Target Variable Completeness**: Zero missing values exist in the target column (`Consumption`).
2. **Exogenous Features**: Wind and solar generation records started becoming systematically tracked in Germany around 2010 and 2012 respectively; unrecorded values prior to these years are cleanly handled or excluded when focusing on consumption forecasting.
3. **Temporal Patterns**:
   - Weekly seasonality: Noticeable drop in industrial/commercial electricity demand during weekends.
   - Annual seasonality: Clear peaks in winter months (heating, indoor lighting, industrial cycles) and troughs in summer months.
   - Trend: Structural transitions in energy efficiency over the 12-year window.

## Custom / Hourly Dataset Compatibility
The pipeline is designed to be modular:
- Any custom CSV with a datetime column (`Date` or `Datetime` or `Timestamp`) and a numerical demand column (`Consumption`, `PJME_MW`, `Demand`, or `Load`) can be placed in this folder or passed to the pipeline.
