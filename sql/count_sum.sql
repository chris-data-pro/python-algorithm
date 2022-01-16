--given 2 tables parents and children,
--give me all parents that have at least one male child and one female child

begin;
CREATE TABLE development.children (
   child_id int,
   child_name varchar(50),
   Child_Gender varchar(50),
   parent_id int
);

INSERT INTO development.children VALUES
(1, 'David','Male', 6),
(2, 'Emma','Male', 4),
(3, 'Alice','Female', 3),
(4, 'David','Male', 3),
(5, 'Tom','Female', 1),
(6, 'Nathen','Male', 2),
(7, 'Akira','Male', 5),
(8, 'Kate','Male', 1),
(9, 'Lily','Female', 3);
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


select * from development.children;
select * from development.parents;







--The SQL NATURAL JOIN is a type of JOIN, it's like INNER JOIN
--It is structured in such a way that, columns with same name of associate tables will appear once only.
--The associated tables have one or more pairs of identically named columns.
--The columns must be the same data type.
--Don’t use ON clause in a natural join.
select * from development.children NATURAL JOIN development.parents order by 1;






select parent_id,
       SUM(CASE WHEN child_gender='Male' THEN 1 ELSE 0 END) as male_ct,
       SUM(CASE WHEN child_gender='Female' THEN 1 ELSE 0 END) as female_ct
from development.children
group by 1
having female_ct >= 1 and male_ct >= 1
order by 1;

select parent_name
from development.children c join development.parents p
on c.parent_id = p.parent_id
group by 1
having SUM(CASE WHEN child_gender='Male' THEN 1 ELSE 0 END) >= 1
and SUM(CASE WHEN child_gender='Female' THEN 1 ELSE 0 END) >= 1;


select parent_name
from development.children c, development.parents p
where c.parent_id = p.parent_id
group by 1
having SUM(CASE WHEN child_gender='Male' THEN 1 ELSE 0 END) >= 1
and SUM(CASE WHEN child_gender='Female' THEN 1 ELSE 0 END) >= 1;


select *
from development.parents p
join (
 select distinct parent_id from development.children where child_gender = 'Male'
) a on p.parent_id = a.parent_id
join (
 select distinct parent_id from development.children where child_gender = 'Female'
) b on p.parent_id = b.parent_id;

select *
from development.parents p
join development.children a on p.parent_id = a.parent_id
join development.children b on p.parent_id = b.parent_id
where a.child_gender = 'Male' and b.child_gender = 'Female'

select parent_id from development.children where child_gender = 'Male'
INTERSECT
select parent_id from development.children where child_gender = 'Female'



select parent_id, count(distinct child_id) as mct
from development.children
where child_gender = 'Male'
group by 1
having count(distinct child_id) > 0

begin;
drop table IF EXISTS development.parents;
commit;


---------------------------------------------------------------------------------
--Summarize each group of tests.

--The table of results should contain four columns:
--name (name of the group),
--all_test_cases (number of tests in the group),
--passed_test_cases (number of test cases with the status OK),
--and total_value (total value of passed tests in this group).

--Rows should be ordered by decreasing total_value.
--In the case of a tie, rows should be sorted lexicographically by name.
---------------------------------------------------------------------------------
begin;
CREATE TABLE development.test_groups (
    name          VARCHAR(40)   NOT NULL,
    test_value    INT   NOT NULL,
    unique(name)
);
commit;

begin;
INSERT INTO development.test_groups VALUES
('performance', 15),
('corner cases', 10),
('numerical stability', 20),
('memory usage', 10);
commit;


begin;
CREATE TABLE development.test_cases (
    id  INTEGER   NOT NULL,
    group_name    VARCHAR(40)   NOT NULL,
    status        VARCHAR(5)    NOT NULL,
    unique(id)
);
commit;

begin;
INSERT INTO development.test_cases VALUES
(13, 'memory usage', 'OK'),
(14, 'numerical stability', 'OK'),
(15, 'memory usage', 'ERROR'),
(16, 'numerical stability', 'OK'),
(17, 'numerical stability', 'OK'),
(18, 'performance', 'ERROR'),
(19, 'performance', 'ERROR'),
(20, 'memory usage', 'OK'),
(21, 'numerical stability', 'OK');
commit;


SELECT tg.name,
       COUNT(tc.id) as all_test_cases,
       SUM(CASE WHEN tc.status = 'OK' THEN 1 ELSE 0 END) as passed_test_cases,
       SUM(CASE WHEN tc.status = 'OK' THEN tg.test_value ELSE 0 END) as total_value
FROM development.test_groups tg
LEFT JOIN development.test_cases tc
ON tg.name = tc.group_name
GROUP BY 1
ORDER BY 4 DESC, 1;





begin;
drop table IF EXISTS development.test_groups;
commit;

