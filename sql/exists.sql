-- EXISTS ( subquery ) Returns TRUE if a subquery contains any rows.

--The following example returns a result set, because EXISTS (SELECT NULL) still evaluates to TRUE
SELECT DepartmentID, Name
FROM HR.Department
WHERE EXISTS (SELECT NULL)
ORDER BY Name ASC ;


SELECT * FROM development.children a WHERE EXISTS (
SELECT 1/0 FROM development.parents b WHERE a.parent_id = b.parent_id)

-- will scan each row in a, treat this row's a.parent_id as a constant, then run the subquery
-- if table b has at lease 1 parent_id equals to this constant, the subquery returns true then return the row in a scanned

-- 扫描a里的每一行，把当前行的a.parent_id看作一个常量，放到subquery里run
-- 如果table b里找不到任何一个parent_id等于这个常量，subquery返回false，不打印当前行，否则打印a里的当前这一行
