SELECT
  COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4)) AS Year,
  SUM(OD.Quantity*OD.UnitPrice*(1-OD.Discount)) AS TotalRevenue
FROM Orders O
JOIN "Order Details" OD ON O.OrderID = OD.OrderID
GROUP BY Year
ORDER BY Year;
