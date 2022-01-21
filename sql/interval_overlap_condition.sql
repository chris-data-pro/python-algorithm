begin;
create table development.user_start_end (
  user_id                        int,
  start_date                    date,
  end_date                      date
);
commit;



begin;
INSERT INTO development.user_start_end
(user_id, start_date, end_date) VALUES
(1, date('2019-01-01'), date('2019-01-31')),
(2, date('2019-01-15'), date('2019-01-17')),
(3, date('2019-01-29'), date('2019-02-04')),
(4, date('2019-02-05'), date('2019-02-10'));
commit;


SELECT * FROM development.user_start_end;


-- self join
SELECT s.user_id as sid, s.start_date as ssd, s.end_date as sed, l.user_id as lid, l.start_date as lsd, l.end_date as led
FROM development.user_start_end s
JOIN development.user_start_end l
ON s.user_id != l.user_id and
(s.end_date between l.start_date and l.end_date OR s.start_date between l.start_date and l.end_date)  -- this is the condition

--sid    ssd                    sed            lid        lsd               led
--2 2019-01-15 00:00:00 2019-01-17 00:00:00     1   2019-01-01 00:00:00 2019-01-31 00:00:00
--3 2019-01-29 00:00:00 2019-02-04 00:00:00     1   2019-01-01 00:00:00 2019-01-31 00:00:00
--1 2019-01-01 00:00:00 2019-01-31 00:00:00     3   2019-01-29 00:00:00 2019-02-04 00:00:00


with base as (
SELECT s.user_id as sid, s.start_date as ssd, s.end_date as sed, l.user_id as lid, l.start_date as lsd, l.end_date as led
FROM development.user_start_end s
JOIN development.user_start_end l
ON s.user_id != l.user_id and
(s.end_date between l.start_date and l.end_date OR s.start_date between l.start_date and l.end_date)),
all_overlapping as (
SELECT sid as uid from base UNION ALL SELECT lid as uid from base)
select a.user_id, CASE WHEN b.uid is NULL THEN 0 ELSE 1 END as overlap
from development.user_start_end a left join all_overlapping b
on a.user_id = b.uid
GROUP by 1, 2
order by 1

--user_id overlap
--    1       1
--    2       1
--    3       1
--    4       0