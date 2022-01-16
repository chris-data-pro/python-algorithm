--SQL Server: returns any distinct values from the query left of EXCEPT that aren't found on the right query.
SELECT ProductID
FROM Production.Product
EXCEPT
SELECT ProductID
FROM Production.WorkOrder ;


--Oracle: the same
SELECT ProductID
FROM Production.Product
MINUS
SELECT ProductID
FROM Production.WorkOrder ;


--Both can be done by left join
SELECT l.ProductID
FROM Production.Product p
LEFT JOIN Production.WorkOrder w
ON p.ProductID = w.ProductID
WHERE w.ProductID IS NULL;


