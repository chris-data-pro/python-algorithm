-- EXISTS ( subquery ) Returns TRUE if a subquery contains any rows.

--The following example returns a result set, because EXISTS (SELECT NULL) still evaluates to TRUE
SELECT DepartmentID, Name
FROM HR.Department
WHERE EXISTS (SELECT NULL)
ORDER BY Name ASC ;


