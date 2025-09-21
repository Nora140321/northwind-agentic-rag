SELECT
  C.CompanyName,
  SUM(OD.Quantity*OD.UnitPrice*(1-OD.Discount)) AS TotalRevenue,
  COUNT(DISTINCT O.OrderID) AS OrderCount
FROM Customers C
JOIN Orders O         ON O.CustomerID = C.CustomerID
JOIN "Order Details" OD ON OD.OrderID   = O.OrderID
WHERE COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4)) = '2012'
GROUP BY C.CompanyName
ORDER BY TotalRevenue DESC
LIMIT 15;
