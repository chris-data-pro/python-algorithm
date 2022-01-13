-----------------------------------------
-- assignment
-----------------------------------------
ACCOUNTS (Columns)

ACCOUNT_ID
ACCOUNT_NAME
SUBSCRIPTION_ID


ACCOUNT_DETAILS (Columns)

ACCOUNT_ID
ACCOUNT_COUNTRY
CSM_MANAGER_NAME


USAGE_DETAILS (Columns)

SUBSCRIPTION_ID
MONTH_ID
LICENSE_QTY
ACTIVE_HOST
TOTAL_MEETINGS


SITE_DETAILS (Columns)

SUBSCRIPTION_ID
SITE_ID
SITE_NAME
AUDIO_TYPE

--4) Show the accounts with subscriptions that have number of “Active Hosts” more than License qty for previous month
with base as (Select DISTINCT month_id, dense_rank() over (order by month_id DESC) as rk
              From "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES")
select a.account_id, b.month_id
from "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" as a
join "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES" as b
on a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
join base
on base.month_id = b.month_id
where base.rk = 2

-- or
with base as (Select DISTINCT SUBSCRIPTION_ID, month_id, dense_rank() over (order by month_id DESC) as rk
              From "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES")
select a.account_id, b.month_id
from "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" as a
join base b
on a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
where b.rk = 2

-- or
select a.account_id, b.month_id, dense_rank() over (order by b.month_id DESC) as rk
from "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" as a
join "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES" as b
on a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
qualify rk = 2


SELECT site_name,
       lag(month_id, 1) over (partition by site_name order by month_id) as last_month,  -- assuming continuous months
       lag(TOT_MTGS, 1) over (partition by site_name order by month_id) as last_tot,
       month_id as current_month, TOT_MTGS as current_tot,
       lag(month_id, -1) over (partition by site_name order by month_id) as next_month,
       lag(TOT_MTGS, -1) over (partition by site_name order by month_id) as next_tot
FROM "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES"
WHERE site_name = 'uson'


-- 选比前月增长的
SELECT site_name,
       lag(TOT_MTGS, 1) over (partition by site_name order by month_id) as last_tot,
       month_id as current_month,
       TOT_MTGS as current_tot
FROM "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES"
WHERE site_name = 'uson'
QUALIFY current_tot > last_tot  -- window function can only be outside SELECT, QUALIFY or ORDER BY clauses


--5) Show the subscriptions with decrease in TOTAL_MEETINGS from previous month to current month
--   with at least one site has Audio Type as “CCA SP”
SELECT SUBSCRIPTION_ID
FROM USAGE_DETAILS ud JOIN SITE_DETAILS sd ON ud.SUBSCRIPTION_ID = sd.SUBSCRIPTION_ID
WHERE sd.AUDIO_TYPE = 'CCA SP'
QUALIFY LAG(ud.TOT_MTGS, 1) OVER (PARTITION BY ud.SUBSCRIPTION_ID ORDER BY ud.MONTH_ID) > TOT_MTGS

