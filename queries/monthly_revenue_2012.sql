SELECT
  COALESCE(STRFTIME('%Y-%m', datetime(Orders.OrderDate)), substr(Orders.OrderDate,1,7)) AS Month,
  SUM("Order Details".Quantity * "Order Details".UnitPrice * (1 - "Order Details".Discount)) AS TotalRevenue
FROM Orders
JOIN "Order Details" ON Orders.OrderID = "Order Details".OrderID
WHERE COALESCE(STRFTIME('%Y', datetime(Orders.OrderDate)), substr(Orders.OrderDate,1,4)) = '2012'
GROUP BY Month
ORDER BY Month;
