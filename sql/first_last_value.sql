-- 10 source number, fist called = last called => 'Y', else => 'N'
begin;
DROP TABLE IF EXISTS development.phone_dial;
commit;

begin;
create table development.phone_dial (
  source_number                  int,
  destination_number             int,
  call_start_date                datetime
);
commit;

SELECT * FROM development.phone_dial;

begin;
INSERT INTO development.phone_dial
(source_number, destination_number, call_start_date) VALUES
(1234, 4567, CONVERT(datetime, '2011-07-20 10:00:00')),
(1234, 2345, CONVERT(datetime, '2011-07-20 11:00:00')),
(1234, 3456, CONVERT(datetime, '2011-07-20 12:00:00')),
(1234, 3456, CONVERT(datetime, '2011-07-20 13:00:00')),
(1234, 4567, CONVERT(datetime, '2011-07-20 15:00:00')),
(1222, 7890, CONVERT(datetime, '2011-07-20 10:00:00')),
(1222, 7680, CONVERT(datetime, '2011-07-20 12:00:00')),
(1222, 2345, CONVERT(datetime, '2011-07-20 13:00:00'));
commit;


SELECT DISTINCT source_number,  --如果不distinct，原table里每一行都输出一行结果
       FIRST_VALUE(destination_number) OVER (PARTITION BY source_number ORDER BY call_start_date rows between unbounded preceding and unbounded following) AS first_called,
       LAST_VALUE(destination_number) OVER (PARTITION BY source_number ORDER BY call_start_date rows between unbounded preceding and unbounded following) AS last_called,
       (CASE WHEN first_called = last_called THEN 'Y' ELSE 'N' END) AS flag
FROM development.phone_dial;


--or
SELECT f.source_number, (CASE WHEN f.first_called = l.last_called THEN 'Y' ELSE 'N' END) AS flag
FROM
(select source_number, destination_number as first_called
from development.phone_dial
qualify rank() over (partition by source_number order by call_start_date) = 1) f
JOIN
(select source_number, destination_number as last_called
from development.phone_dial
qualify rank() over (partition by source_number order by call_start_date desc) = 1) l
ON f.source_number = l.source_number;


--source_number    flag
--1222              N
--1234              Y


--or
with asc_order as (
select source_number, destination_number as first_called from (
select source_number, destination_number,
rank() over (partition by source_number order by call_start_date) as rk
from development.phone_dial) a
where a.rk = 1),
desc_order as (
select source_number, destination_number as last_called from (
select source_number, destination_number,
rank() over (partition by source_number order by call_start_date desc) as rk
from development.phone_dial) b
where b.rk = 1)
select ao.source_number, (case when ao.first_called = deo.last_called then 'Y' else 'N' end) as flag
from asc_order ao join desc_order deo on ao.source_number = deo.source_number;




-- 5) identify callers who made their first call and the last call to the same recipient ON A GIVEN DAY
begin;
DROP TABLE IF EXISTS development.phone_log;
commit;

begin;
create table development.phone_log (
  caller_id                  int,
  recipient_id               int,
  call_start_time            datetime
);
commit;

-- SELECT * FROM development.phone_log;

begin;
INSERT INTO development.phone_log
(caller_id, recipient_id, call_start_time) VALUES
(101, 201, CONVERT(datetime, '2015-08-31 02:35:00')),
(101, 301, CONVERT(datetime, '2015-08-31 02:37:00')),
(101, 501, CONVERT(datetime, '2015-08-31 02:39:00')),
(101, 201, CONVERT(datetime, '2015-09-01 02:41:00')),
(101, 201, CONVERT(datetime, '2015-09-01 05:00:00')),
(103, 401, CONVERT(datetime, '2015-08-31 05:35:00')),
(104, 501, CONVERT(datetime, '2015-08-31 06:35:00')),
(104, 601, CONVERT(datetime, '2015-08-31 07:35:00')),
(104, 601, CONVERT(datetime, '2015-09-01 02:01:00')),
(104, 201, CONVERT(datetime, '2015-09-01 03:01:00')),
(104, 601, CONVERT(datetime, '2015-09-01 05:30:00')),
(105, 201, CONVERT(datetime, '2015-09-01 09:00:00'));
commit;

with base as
(SELECT caller_id,
       date(call_start_time) as day,
       FIRST_VALUE(recipient_id) OVER (PARTITION BY caller_id, day ORDER BY call_start_time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as first_called,
       LAST_VALUE(recipient_id) OVER (PARTITION BY caller_id, day ORDER BY call_start_time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_called
FROM development.phone_log)
select day, caller_id, first_called, last_called
from base
where first_called = last_called
group by 1, 2, 3, 4  -- 其实就是想选 distinct day, caller_id, first_called, last_called
order by 1, 2;


--day                          caller_id   first_called   last_called
--2015-08-31 00:00:00               103          401         401
--2015-09-01 00:00:00               101          201         201
--2015-09-01 00:00:00               104          601         601
--2015-09-01 00:00:00               105          201         201


--or
with asc_order as (
select day, caller_id, recipient_id
from
(
select cast(call_start_time as date) as day, caller_id, recipient_id, rank() over (partition by day, caller_id order by call_start_time) as rk
from development.phone_log
) foo
where foo.rk = 1),
desc_order as (
select day, caller_id, recipient_id
from
(
select cast(call_start_time as date) as day, caller_id, recipient_id, rank() over (partition by day, caller_id order by call_start_time desc) as rk
from development.phone_log
) b
where b.rk = 1)
select ao.day as call_date, ao.caller_id, ao.recipient_id
from asc_order ao
join desc_order deo
on ao.day = deo.day and ao.caller_id = deo.caller_id and ao.recipient_id = deo.recipient_id
order by 1, 2;


--call_date                    caller_id  recipient_id
--2015-08-31 00:00:00               103      401
--2015-09-01 00:00:00               101      201
--2015-09-01 00:00:00               104      601
--2015-09-01 00:00:00               105      201

