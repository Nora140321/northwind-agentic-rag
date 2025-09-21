WITH rev_by_month AS (
  SELECT
    COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4)) AS Year,
    COALESCE(STRFTIME('%m', datetime(O.OrderDate)), substr(O.OrderDate,6,2)) AS MonthNum,
    SUM(OD.Quantity * OD.UnitPrice * (1 - OD.Discount)) AS TotalRevenue
  FROM Orders O
  JOIN "Order Details" OD ON O.OrderID = OD.OrderID
  GROUP BY Year, MonthNum
)
SELECT
  CAST(MonthNum AS INT) AS MonthNumber,
  ROUND(AVG(TotalRevenue), 2) AS AvgRevenue
FROM rev_by_month
GROUP BY MonthNum
ORDER BY CAST(MonthNum AS INT);
