--// 加一列
ALTER TABLE batch_assignment_request
ADD COLUMN processedCount bigint(20);

-- 删一列
ALTER TABLE Customers
DROP COLUMN ContactName;
