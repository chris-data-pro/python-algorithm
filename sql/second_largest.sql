begin;
DROP  TABLE IF EXISTS development.parents;
commit;

begin;
CREATE TABLE development.parents (
   parent_id int,
   parent_name varchar(50)
);

INSERT INTO development.parents VALUES
(1, 'p1'),
(2, 'p2'),
(3, 'p3'),
(5, 'p5'),
(6, 'p6');
commit;

select max(parent_id)
from development.parents
where parent_id < (SELECT MAX(parent_id) FROM development.parents)

with base as (
select *, row_number() over (order by parent_id desc) as rn
from development.parents)
select * from base where rn = 2

select *, row_number() over (order by parent_id desc) as rn
from development.parents
qualify row_number() over (order by parent_id desc) = 2
