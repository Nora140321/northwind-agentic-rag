## Schema Patterns and Conventions

### Revenue formula
- Always compute revenue as: `Quantity * UnitPrice * (1 - Discount)`

### Average Order Value (AOV)
- To get true AOV, compute per-order totals first, then average:
  1. CTE grouped by `(Orders.OrderID, Orders.CustomerID)` with `SUM(Quantity * UnitPrice * (1 - Discount)) AS OrderValue`
  2. Aggregate per customer (or shipper, etc.) with `AVG(OrderValue)`
- Optional validation column: `COUNT(*) AS OrdersCount`

### Geography
- Customer country: `Customers.Country`
- Shipment destination country: `Orders.ShipCountry` (not `Customers.Country`)
- Shipper/carrier name: join `Orders.ShipVia = Shippers.ShipperID`
- Use `Shippers.CompanyName AS ShipperName` as the canonical output alias

### Time bucketing
- Year: `STRFTIME('%Y', datetime(Orders.OrderDate)) AS Year`
- Month: `STRFTIME('%Y-%m', datetime(Orders.OrderDate)) AS Month`
- Quarter: `((CAST(STRFTIME('%m', datetime(Orders.OrderDate)) AS INTEGER) - 1) / 3) + 1 AS Quarter`

### Category and product joins
- `Order Details` → Products via `ProductID`
- Products → Categories via `CategoryID`
- Use these joins for category-level revenue aggregations

### Category revenue share pattern (example 2016)
```sql  
WITH CategoryRevenue AS (  
  SELECT  
    c.CategoryName,  
    SUM(od.Quantity * od.UnitPrice * (1 - od.Discount)) AS TotalRevenue  
  FROM "Order Details" od  
  JOIN Products p   ON od.ProductID = p.ProductID  
  JOIN Categories c ON p.CategoryID = c.CategoryID  
  JOIN Orders o     ON od.OrderID = o.OrderID  
  WHERE STRFTIME('%Y', datetime(o.OrderDate)) = '2016'  
  GROUP BY c.CategoryName  
),  
Total AS (  
  SELECT SUM(TotalRevenue) AS GrandTotal FROM CategoryRevenue  
)  
SELECT  
  cr.CategoryName,  
  ROUND(100.0 * cr.TotalRevenue / t.GrandTotal, 2) AS RevenueSharePct  
FROM CategoryRevenue cr  
CROSS JOIN Total t  
ORDER BY RevenueSharePct DESC;  