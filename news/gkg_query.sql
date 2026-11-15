-- Headlines about USD / EUR / JPY / SGD from GDELT's Global Knowledge Graph.
-- Paste into the BigQuery console (free sandbox is enough), check the "This query will
-- process N GB" estimate top right, run, then Save results -> CSV, save the file in
-- data/news/ and load it with:  python -m news.bigquery data/news/<file>.csv
--
-- Change the dates to add more days (e.g. the last week, for the live server and replay).
DECLARE start_day DATE DEFAULT '2026-07-01';
DECLARE end_day   DATE DEFAULT '2026-09-01';   -- exclusive

SELECT
  CAST(DATE AS STRING) AS date,                -- YYYYMMDDHHMMSS, UTC, when GDELT saw it
  SourceCommonName AS domain,
  DocumentIdentifier AS url,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP(start_day)
  AND _PARTITIONTIME <  TIMESTAMP(DATE_ADD(end_day, INTERVAL 1 DAY))  -- partitions can lag a little
  AND DATE >= CAST(FORMAT_DATE('%Y%m%d000000', start_day) AS INT64)
  AND DATE <  CAST(FORMAT_DATE('%Y%m%d000000', end_day) AS INT64)
  AND (TranslationInfo IS NULL OR TranslationInfo = '')               -- English-language sources
  AND REGEXP_CONTAINS(
        REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>'),
        r'(?i)\b(dollars?|greenback|federal reserve|fomc|euros?|ecb|eurozone|euro area|lagarde|yen|bank of japan|boj|singapore dollar|monetary authority of singapore)\b')
