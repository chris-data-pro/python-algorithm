-- running sum or running average

SELECT Id, StudentName, StudentGender, StudentAge,
SUM (StudentAge) OVER (PARTITION BY StudentGender ORDER BY Id) AS RunningAgeTotal,
AVG (StudentAge) OVER (PARTITION BY StudentGender ORDER BY Id) AS RunningAgeAverage
FROM Students


begin;
DROP TABLE IF EXISTS development.orders;
commit;

begin;
create table development.orders (
  order_id                        int,
  cust_id                         int,
  order_date                      date,
  product                         VARCHAR(20),
  order_amount                    int

);
commit;

SELECT * FROM development.orders;

begin;
INSERT INTO development.orders
(order_id, cust_id, order_date, product, order_amount) VALUES
(50001, 101, date('2015-08-29'), 'Camera', 100),
(50002, 102, date('2015-08-30'), 'Shoes', 90),
(50003, 103, date('2015-05-31'), 'Laptop', 400),
(50004, 101, date('2015-08-29'), 'Mobile', 100),
(50005, 104, date('2015-08-29'), 'FrozenMeals', 30),
(50006, 104, date('2015-08-30'), 'Cloths', 65);
commit;


SELECT order_date, order_id,
       SUM(order_amount) OVER (PARTITION by order_date ORDER BY order_id rows unbounded preceding) as moving_sum,
       AVG(order_amount) OVER (PARTITION by order_date ORDER BY order_id rows unbounded preceding) as moving_average
from development.orders
order by 1, 2


2015-05-31 00:00:00 50003 400 400
2015-08-29 00:00:00 50001 100 100
2015-08-29 00:00:00 50004 200 100
2015-08-29 00:00:00 50005 230 76
2015-08-30 00:00:00 50002 90 90
2015-08-30 00:00:00 50006 155 77