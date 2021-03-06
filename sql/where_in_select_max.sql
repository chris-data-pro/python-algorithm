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


-- How many accounts have subscriptions with more than 1000 licenses in the latest month (current month)
SELECT a.ACCOUNT_ID, b.SUBSCRIPTION_ID, b.month_id, b.TOT_LICENSES
FROM "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" a
JOIN "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES" b
ON a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
WHERE b.TOT_LICENSES > 1000 AND
b.month_id in 
(Select max(month_id) From "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES") --整个table选一个
order by 4;

-- or

SELECT a.ACCOUNT_ID, b.SUBSCRIPTION_ID, b.month_id, b.TOT_LICENSES
FROM "COLLAB_DB"."COLLAB_DS2_CSWI"."CS_SKU_BLIS_ANNUITY" a
JOIN "COLLAB_DB"."COLLAB_DS2_CSSTG"."PM_WBX_SITE_USAGE_MLY_WITH_PREDICTIVE_FEATURES" b
ON a.SUBSCRIPTION_ID = b.SUBSCRIPTION_ID
WHERE b.TOT_LICENSES > 1000
QUALIFY rank() over (order by b.month_id DESC) = 1
order by 4;


SELECT id, hashId
FROM experiment
WHERE status = 'active'
AND id IN (SELECT MAX(id) FROM experiment GROUP BY hashId) --选出一组ids
;