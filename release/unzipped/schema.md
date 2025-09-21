# Northwind — Schema Notes and Business Rules (for NL→SQL)

These notes ground the Planner agent to generate correct SQLite SQL against the Northwind database.

## Core Tables (key columns)
- Customers(CustomerID, CompanyName, Country, City, Region, PostalCode, Address, Phone, Fax)
- Orders(OrderID, CustomerID, EmployeeID, OrderDate, ShipVia, Freight, ShipName, ShipCountry, ShipCity)
- "Order Details"(OrderID, ProductID, UnitPrice, Quantity, Discount)
- Products(ProductID, ProductName, CategoryID, SupplierID, UnitPrice, Discontinued)
- Categories(CategoryID, CategoryName, Description)
- Employees(EmployeeID, LastName, FirstName, Title, Country, City)
- Shippers(ShipperID, CompanyName)

## Canonical Joins
- Orders.OrderID = "Order Details".OrderID
- "Order Details".ProductID = Products.ProductID
- Products.CategoryID = Categories.CategoryID
- Orders.CustomerID = Customers.CustomerID
- Orders.EmployeeID = Employees.EmployeeID
- Orders.ShipVia = Shippers.ShipperID

## Revenue Measure (TotalRevenue)
Use this exact formula whenever “revenue”, “sales”, or “total sales” is requested:
- TotalRevenue = SUM("Order Details".Quantity × "Order Details".UnitPrice × (1 − "Order Details".Discount))

Example fragment:

## Date Handling (SQLite)
Always cast OrderDate to a datetime for STRFTIME grouping/filtering:
- Month key: STRFTIME('%Y-%m', datetime(Orders.OrderDate)) AS Month
- Year key:  STRFTIME('%Y',   datetime(Orders.OrderDate)) AS Year

Quarter index (returns 1–4 only):

## Geography
- When grouping by country for customers, use Customers.Country (NOT Customers.CompanyName).
- For shipping carrier names, join Orders.ShipVia → Shippers.ShipperID.

## Aggregation Patterns (avoid errors)
- Do NOT nest aggregates like AVG(SUM(...)).
- For "average order value per customer", compute order totals first, then average:

## Common Groupings and Columns
- Products.ProductName
- Categories.CategoryName
- Customers.Country, Customers.CompanyName
- Employees.FirstName || ' ' || Employees.LastName AS Employee
- STRFTIME('%Y', datetime(Orders.OrderDate)) AS Year
- STRFTIME('%Y-%m', datetime(Orders.OrderDate)) AS Month
- Quarter as defined above

## Example Intents (guidance, not strict SQL)
- "Monthly revenue totals for 2016. Columns Month, TotalRevenue. Order by Month."
- "Top 10 customers by total revenue in 2016. Columns CustomerID, CompanyName, TotalRevenue. Order by TotalRevenue desc. Include LIMIT 10."
- "Revenue by category by quarter for 2016. Columns Quarter, CategoryName, TotalRevenue. Order by Quarter, TotalRevenue desc."
- "Best-selling products by quantity overall. Columns ProductName, TotalQuantity. Order by TotalQuantity desc. Include LIMIT 10."
- "Average order value by shipper. Columns ShipperName, AvgOrderValue. Order by AvgOrderValue desc."
- "Total revenue by year. Columns Year, TotalRevenue. Order by Year."

## Notes
- Always use the exact table/column names as above.
- Use INNER JOIN/JOIN unless a left join is explicitly required by the question.
- Unless the user explicitly asks for LIMIT, omit it.
