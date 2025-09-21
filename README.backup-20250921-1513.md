

\# Results Highlights



This demo shows an agentic NL→SQL workflow over the Northwind SQLite database with retrieval‑grounded schema rules.



\## Grounding

\- Canonical schema notes: `docs/schema/schema.md`

\- Revenue measure (line item): `Quantity \* UnitPrice \* (1 - Discount)`

\- All revenue aggregations: `SUM(...)` at the requested grain



\## Key Insight — Yearly Revenue

Artifacts:

\- PNG: `out/revenue\_by\_year.png`  

\- CSV: `out/revenue\_by\_year.csv`  

\- Markdown: `out/revenue\_by\_year.md`  

\- Rounded to 2 decimals (nice for charts):  

&nbsp; - PNG: `out/revenue\_by\_year\_round2.png`  

&nbsp; - CSV: `out/revenue\_by\_year\_round2.csv`  

&nbsp; - Markdown: `out/revenue\_by\_year\_round2.md`



Trend (2012–2023): steady growth into mid‑decade, followed by modest variation; 2015 is the peak in this dataset.



\## Reproduce Locally



Rebuild the vector index (safe if it already exists):

```powershell

Remove-Item -Recurse -Force "data\\vectorstore\_lc" -ErrorAction SilentlyContinue

```



Run the yearly revenue query:

```powershell

python .\\two\_agent\_langchain\_app.py --q "Total revenue by year. Columns Year, TotalRevenue. Order by Year."

```



Save artifacts (non‑rounded):

```powershell

python .\\two\_agent\_langchain\_app.py `

&nbsp; --q "Total revenue by year. Columns Year, TotalRevenue. Order by Year." `

&nbsp; --csv out\\revenue\_by\_year.csv `

&nbsp; --md  out\\revenue\_by\_year.md `

&nbsp; --plot out\\revenue\_by\_year.png `

&nbsp; --plot-kind line

```



Save artifacts (rounded to 2 decimals):

```powershell

python .\\two\_agent\_langchain\_app.py `

&nbsp; --q "Total revenue by year. Use ROUND(SUM(\[Order Details].Quantity \* \[Order Details].UnitPrice \* (1 - \[Order Details].Discount)), 2) AS TotalRevenue. Columns Year, TotalRevenue. Order by Year." `

&nbsp; --csv out\\revenue\_by\_year\_round2.csv `

&nbsp; --md  out\\revenue\_by\_year\_round2.md `

&nbsp; --plot out\\revenue\_by\_year\_round2.png `

&nbsp; --plot-kind line

```



\## Packaging



Create the release zip and validate its contents:

```powershell

Remove-Item .\\release\\northwind-demo.zip -ErrorAction SilentlyContinue

Compress-Archive -Force -DestinationPath .\\release\\northwind-demo.zip `

&nbsp; -Path .\\out, .\\docs, .\\two\_agent\_langchain\_app.py, .\\data\\.env.ini, .\\README.md



Expand-Archive -Path .\\release\\northwind-demo.zip -DestinationPath .\\release\\unzipped -Force

Get-ChildItem .\\release\\unzipped\\out -File | Measure-Object

```



\## Notes

\- If you hit Gemini free‑tier 429s, retry after the daily reset, switch to `gemini-1.5-flash-8b`, or set an OpenAI fallback in `data/.env.ini`.

\- Keep `SCHEMA\_DIR=docs\\schema` and treat `data\\docs\\schema\\\_archive` as read‑only.



