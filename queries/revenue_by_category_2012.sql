SELECT
  C.CategoryName,
  SUM(OD.Quantity*OD.UnitPrice*(1-OD.Discount)) AS TotalRevenue
FROM Categories C
JOIN Products P       ON P.CategoryID=C.CategoryID
JOIN "Order Details" OD ON OD.ProductID=P.ProductID
JOIN Orders O         ON O.OrderID=OD.OrderID
WHERE COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4))='2012'
GROUP BY C.CategoryName
ORDER BY TotalRevenue DESC;
