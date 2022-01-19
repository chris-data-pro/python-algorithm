--// 查看date
SELECT metricId,
       CAST(date AS CHAR) AS day,
       COUNT(*)
FROM KpiMetricValue
WHERE metricId IN (29, 30, 31) GROUP BY 1,2;

SELECT id, experimentName, DATE_FORMAT(dumpDate, '%Y-%m-%d'), DATE_FORMAT(endDate, '%Y-%m-%d')
FROM phoenix_results_window_v2
ORDER BY RAND()
limit 10;

SELECT metricId, dimensionId, CAST(date AS CHAR) AS day, state, createdAt, debugInfo
FROM KpiMetricAnomaly
WHERE metricId = 6;

SELECT DISTINCT experimentName, DATE_FORMAT(endDate,'%d/%m/%Y') as data_through_date
FROM phoenix_results_window_v2
where endDate > '2021-09-09';

-- to date
SELECT datefromparts(2020, 07, 25), datefromparts('2020', '07', '25'), '2021-01-31'::date, date('2021-01-31')


-- PostgreSQL


--to date
SELECT '2021-01-31'::date, date('2021-01-31')


begin;
drop table IF EXISTS development.games;
commit;

begin;
CREATE TABLE development.games (
   home_team varchar(1),
   home_score int,
   away_team varchar(1),
   away_score int,
   game_date date
);
commit;

begin;
INSERT INTO development.games VALUES
('A', 3, 'B', 2, date('2021-08-01')),
('C', 0, 'D', 0, date('2021-08-02')),
('A', 3, 'D', 1, date('2021-08-12')),
('C', 1, 'B', 2, date('2021-08-12')),
('A', 0, 'C', 2, date('2021-08-22')),
('B', 4, 'D', 1, date('2021-08-23')),
('B', 0, 'A', 5, date('2021-09-01')),
('D', 0, 'C', 3, date('2021-09-02')),
('D', 0, 'A', 1, date('2021-09-12')),
('B', 0, 'C', 2, date('2021-09-12')),
('C', 0, 'A', 1, date('2021-09-22')),
('D', 2, 'B', 2, date('2021-09-23'));
commit;


SELECT game_date，date(game_date + interval '1 day') as next_date,
       EXTRACT(YEAR FROM game_date), EXTRACT(MONTH FROM game_date), EXTRACT(DAY FROM game_date),
       EXTRACT(DOW FROM game_date), EXTRACT(DOY FROM game_date), EXTRACT(QUARTER FROM game_date)
FROM development.games;


-- assignment_st's type: timestamp
SELECT assignment_st,
       EXTRACT(YEAR FROM assignment_st), EXTRACT(MONTH FROM assignment_st), EXTRACT(DAY FROM assignment_st),
       EXTRACT(DOW FROM assignment_st), EXTRACT(DOY FROM assignment_st), EXTRACT(QUARTER FROM assignment_st),
       EXTRACT(HOUR FROM assignment_st), EXTRACT(MINUTE FROM assignment_st), EXTRACT(SECOND FROM assignment_st)
FROM analytics_rollup.experiment_users
LIMIT 3;


-- day's type: date
SELECT EXTRACT(YEAR FROM day), DATE_PART('year', day),
       EXTRACT(MONTH FROM day), DATE_PART('month', day),
       EXTRACT(DAY FROM day), DATE_PART('day', day),
       day, date(day + interval '1 day') as next
FROM analytics_rollup_finance.cashflow_uid_day
LIMIT 3

-- assignment_st's type: timestamp
SELECT assignment_st,
       EXTRACT(YEAR FROM assignment_st), EXTRACT(MONTH FROM assignment_st), EXTRACT(DAY FROM assignment_st),
       DATE_PART('year', assignment_st), DATE_PART('month', assignment_st), DATE_PART('day', assignment_st),
       EXTRACT(DOW FROM assignment_st), EXTRACT(DOY FROM assignment_st), EXTRACT(QUARTER FROM assignment_st),
       DATE_PART('dow', assignment_st), DATE_PART('doy', assignment_st), DATE_PART('quarter', assignment_st),
       EXTRACT(HOUR FROM assignment_st), EXTRACT(MINUTE FROM assignment_st), EXTRACT(SECOND FROM assignment_st),
       DATE_PART('hour', assignment_st), DATE_PART('minute', assignment_st), DATE_PART('second', assignment_st)
FROM analytics_rollup.experiment_users
LIMIT 3



-- 11 given historical latest order customers, and new daily snapshot order customers
-- get new customers and reactivated (last order was more than 1 year back) customers
begin;
DROP TABLE IF EXISTS development.customer_orders;
commit;

begin;
create table development.customer_orders (
  order_day                       date,
  customer_id                     int
);
commit;

SELECT * FROM development.customer_orders;

begin;
INSERT INTO development.customer_orders
(order_day, customer_id) VALUES
(date('2019-07-20'), 1),
(date('2020-09-20'), 2),
(date('2021-07-20'), 3);
commit;

begin;
DROP TABLE IF EXISTS development.customer_orders_daily;
commit;

begin;
create table development.customer_orders_daily (
  snapshot_day                    date,
  order_day                       date,
  customer_id                     int
);
commit;

SELECT * FROM development.customer_orders_daily;

begin;
INSERT INTO development.customer_orders_daily
(snapshot_day, order_day, customer_id) VALUES
(date('2020-10-11'), date('2020-10-11'), 1),
(date('2020-10-11'), date('2020-10-11'), 2),
(date('2020-10-11'), date('2020-10-11'), 4),
(date('2020-10-11'), date('2020-10-11'), 5);
commit;


SELECT n.snapshot_day,
       SUM(CASE WHEN h.customer_id IS NULL THEN 1 ELSE 0 END) AS new_custormers,
       SUM(CASE WHEN n.order_day - interval '365 days' > h.order_day THEN 1 ELSE 0 END) AS reactivated_customers
FROM development.customer_orders_daily n
LEFT JOIN development.customer_orders h
ON n.customer_id = h.customer_id
GROUP BY 1

--snapshot_day        new_customers    reactivated_customers
--2020-10-11 00:00:00         2               1


-- all reactivated customers
SELECT n.customer_id
FROM development.customer_orders_daily n
JOIN development.customer_orders h
ON n.customer_id = h.customer_id
WHERE n.order_day - interval '365 days' > h.order_day


-- all new customers
SELECT customer_id FROM development.customer_orders_daily
MINUS
SELECT n.customer_id FROM development.customer_orders


