WITH m AS (
  SELECT
    COALESCE(STRFTIME('%Y-%m', datetime(O.OrderDate)), substr(O.OrderDate,1,7)) AS ym,
    SUM(OD.Quantity*OD.UnitPrice*(1-OD.Discount)) AS rev
  FROM Orders O
  JOIN "Order Details" OD ON O.OrderID=OD.OrderID
  WHERE COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4))='2012'
  GROUP BY ym
)
SELECT
  (SELECT SUM(rev) FROM m) AS SumOfMonths,
  (SELECT SUM(OD.Quantity*OD.UnitPrice*(1-OD.Discount))
   FROM Orders O
   JOIN "Order Details" OD ON O.OrderID=OD.OrderID
   WHERE COALESCE(STRFTIME('%Y', datetime(O.OrderDate)), substr(O.OrderDate,1,4))='2012') AS AnnualTotal;
