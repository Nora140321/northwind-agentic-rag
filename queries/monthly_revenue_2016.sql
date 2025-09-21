SELECT
  COALESCE(STRFTIME('%Y-%m', datetime(O.OrderDate)), substr(O.OrderDate,1,7)) AS Month,
  SUM(OD.Quantity * OD.UnitPrice * (1 - OD.Discount)) AS TotalRevenue
FROM Orders O
JOIN "Order Details" OD ON O.OrderID = OD.OrderID
WHERE COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4)) = '2016'
GROUP BY Month
ORDER BY Month;
