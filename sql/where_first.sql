-- SELECT * FROM TABLE WHERE ...
-- firstly evaluate where condition

--tableA

student_name

Alice
Tom
Chris


select 1 from tableA WHERE false
-- returns 0 row

select 1 from development.parents where true
select 1 from tableA WHERE true -- every row will be selected first, and then select 1 from each row
-- returns

1
1
1


-- the SELECT in EXISTS subquery is ignored, divided by 0 is ok
SELECT *
FROM SUPPLIERS s
WHERE EXISTS (SELECT 1/0
              FROM ORDERS o
              WHERE o.supplier_id = s.supplier_id)

-- will scan each row in s, treat this row's s.supplier_id as a constant, then run the subquery
-- if table o has at lease 1 supplier_id equals to this constant, the subquery returns true then return the row in s scanned


SELECT *
FROM SUPPLIERS s
WHERE NOT EXISTS (SELECT 1/0
                  FROM ORDERS o
                  WHERE o.supplier_id = s.supplier_id)

-- 扫描s里的每一行，把当前行的s.supplier_id看作一个常量，放到subquery里run
-- 如果table o里找不到任何一个supplier_id等于这个常量，subquery返回false，不打印当前行，否则打印s里的当前这一行
