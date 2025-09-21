WITH rev_by_year AS (
  SELECT
    COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4)) AS Year,
    SUM(OD.Quantity * OD.UnitPrice * (1 - OD.Discount)) AS TotalRevenue
  FROM Orders O
  JOIN "Order Details" OD ON O.OrderID = OD.OrderID
  GROUP BY Year
)
SELECT
  a.Year,
  a.TotalRevenue,
  b.TotalRevenue AS PrevYearRevenue,
  CASE
    WHEN b.TotalRevenue IS NULL OR b.TotalRevenue = 0 THEN NULL
    ELSE ROUND(100.0 * (a.TotalRevenue - b.TotalRevenue) / b.TotalRevenue, 2)
  END AS YoY_Pct
FROM rev_by_year a
LEFT JOIN rev_by_year b ON CAST(b.Year AS INT) = CAST(a.Year AS INT) - 1
ORDER BY CAST(a.Year AS INT);
