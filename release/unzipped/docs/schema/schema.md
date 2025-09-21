\# Northwind — Schema Notes and Business Rules (for NL→SQL)



These notes ground the Planner agent to generate correct SQLite SQL against the Northwind database.



\## Core Tables (key columns)



\- Customers(CustomerID, CompanyName, ContactName, Country, City, Address, Region, PostalCode, Phone, Fax)

\- Orders(OrderID, CustomerID, EmployeeID, OrderDate, RequiredDate, ShippedDate, ShipVia, Freight, ShipName, ShipCountry, ShipCity)

\- "Order Details"(OrderID, ProductID, UnitPrice, Quantity, Discount)

\- Products(ProductID, ProductName, CategoryID, SupplierID, UnitPrice, Discontinued)

\- Categories(CategoryID, CategoryName, Description)

\- Employees(EmployeeID, LastName, FirstName, Title, TitleOfCourtesy)

\- Shippers(ShipperID, CompanyName, Phone)



\## Canonical Joins



\- Orders ↔ "Order Details": `Orders.OrderID = "Order Details".OrderID`

\- "Order Details" ↔ Products: `"Order Details".ProductID = Products.ProductID`

\- Products ↔ Categories: `Products.CategoryID = Categories.CategoryID`

\- Orders ↔ Customers: `Orders.CustomerID = Customers.CustomerID`

\- Orders ↔ Shippers: `Orders.ShipVia = Shippers.ShipperID`



\## Revenue Measure (TotalRevenue)



Use this exact formula whenever “revenue”, “sales”, or “total sales” is requested:



\- Line item: `Quantity \* UnitPrice \* (1 - Discount)`

\- Aggregated: `SUM("Order Details".Quantity \* "Order Details".UnitPrice \* (1 - "Order Details".Discount)) AS TotalRevenue`



\## Date Handling (SQLite)



Always cast `OrderDate` to a datetime for time grouping/filtering.



\- Year: `STRFTIME('%Y', datetime(Orders.OrderDate)) AS Year`

\- Month key: `STRFTIME('%Y-%m', datetime(Orders.OrderDate)) AS Month`

\- Quarter: `((CAST(STRFTIME('%m', datetime(Orders.OrderDate)) AS INTEGER) - 1) / 3) + 1 AS Quarter`  

&nbsp; Ensures quarters are \\(1\\)–\\(4\\) only.



\## Geography and Entity Fields



\- Customer country: `Customers.Country`

\- Shipment destination: `Orders.ShipCountry`

\- Shipper name: `Shippers.CompanyName AS ShipperName`

\- Prefer clear human‑readable aliases: `TotalRevenue`, `AvgOrderValue`, `OrdersCount`, etc.



\## Patterns and Archetypes



\### 1) Countries by Total Revenue (Top N)

```sql  

SELECT  

&nbsp; c.Country AS Country,  

&nbsp; SUM(od.Quantity \* od.UnitPrice \* (1 - od.Discount)) AS TotalRevenue  

FROM Customers c  

JOIN Orders o ON c.CustomerID = o.CustomerID  

JOIN "Order Details" od ON o.OrderID = od.OrderID  

GROUP BY c.Country  

ORDER BY TotalRevenue DESC  

LIMIT 5;  

