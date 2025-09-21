WITH base AS (
  SELECT
    C.CompanyName,
    SUM(OD.Quantity*OD.UnitPrice*(1-OD.Discount)) AS Revenue
  FROM Customers C
  JOIN Orders O         ON O.CustomerID = C.CustomerID
  JOIN "Order Details" OD ON OD.OrderID   = O.OrderID
  WHERE COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4)) = '2012'
  GROUP BY C.CompanyName
),
t10 AS (
  SELECT SUM(Revenue) AS Top10Revenue
  FROM (
    SELECT Revenue
    FROM base
    ORDER BY Revenue DESC
    LIMIT 10
  )
),
tot AS (
  SELECT SUM(Revenue) AS TotalRevenue FROM base
)
SELECT
  tot.TotalRevenue,
  t10.Top10Revenue,
  ROUND(100.0 * t10.Top10Revenue / tot.TotalRevenue, 2) AS Top10_Share_Pct
FROM tot, t10;
