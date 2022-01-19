-- 能不用DISTINCT就不用

SELECT site_name
FROM "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES"
GROUP BY 1


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
