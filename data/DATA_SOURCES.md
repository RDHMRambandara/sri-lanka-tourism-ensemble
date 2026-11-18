# Data sources

`tourism_monthly_2010_2026.csv` has 199 monthly rows (January 2010 to July 2026) and three columns:

| Column | Meaning | Source |
|---|---|---|
| `date` | First day of the month | |
| `arrivals` | International tourist arrivals to Sri Lanka in that month | Sri Lanka Tourism Development Authority (SLTDA) |
| `usd_lkr` | Monthly average spot exchange rate, Sri Lankan rupees per US dollar | Central Bank of Sri Lanka (CBSL) |

## How the arrivals series was built
Monthly tables were read from SLTDA reports (www.sltda.gov.lk):
- Annual Statistical Report 2019 edition: monthly arrivals 2010 to 2019
- Annual Statistical Report (covering 2020): 2020
- Monthly Tourist Arrivals Report, December 2022: 2021 and 2022
- Annual Statistical Report 2024: 2023
- Monthly Tourist Arrivals Report, December 2025: 2024 and 2025
- Monthly Tourist Arrivals Report, July 2026: January to July 2026

## Validation
For every year, the sum of the 12 monthly values equals the yearly total printed in the SLTDA reports. One value (December 2020 = 393) was read from a second table of the same report.

Known source inconsistency: March 2018 is 233,382 in the 2019 annual report and 233,282 in a later report. 233,382 was used because the year then adds up to the official total.

## Exchange rate
CBSL "Monthly average spot exchange rates" spreadsheet (www.cbsl.gov.lk, Rates and Indicators, Exchange Rates).
