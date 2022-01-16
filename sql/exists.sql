-- EXISTS ( subquery ) Returns TRUE if a subquery contains any rows.

--The following example returns a result set, because EXISTS (SELECT NULL) still evaluates to TRUE
SELECT DepartmentID, Name
FROM HR.Department
WHERE EXISTS (SELECT NULL)
ORDER BY Name ASC ;


SELECT * FROM development.children a WHERE EXISTS (
SELECT 1/0 FROM development.parents b WHERE a.parent_id = b.parent_id)