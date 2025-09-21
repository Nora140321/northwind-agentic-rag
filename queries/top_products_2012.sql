SELECT
  P.ProductName,
  SUM(OD.Quantity*OD.UnitPrice*(1-OD.Discount)) AS TotalRevenue
FROM "Order Details" OD
JOIN Orders O   ON O.OrderID=OD.OrderID
JOIN Products P ON P.ProductID=OD.ProductID
WHERE COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4))='2012'
GROUP BY P.ProductName
ORDER BY TotalRevenue DESC
LIMIT 10;
