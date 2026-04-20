-- Reconstructed 7-day windows produced by the imputer.
-- Mirrors the input table conso_historiques_clean structure, with one
-- quality column per building so the 5 nightly per-building runs do not
-- overwrite each other's flags.
-- Apply once per cluster before enabling --source=cassandra writes.

CREATE TABLE IF NOT EXISTS previsions_data.conso_historiques_reconstructed (
    name              text,
    "Date"            timestamp,
    "Ptot_HA"         double,
    "Ptot_HEI_13RT"   double,
    "Ptot_HEI_5RNS"   double,
    "Ptot_RIZOMM"     double,
    "Ptot_Campus"     double,
    quality_ha        int,
    quality_hei_13rt  int,
    quality_hei_5rns  int,
    quality_rizomm    int,
    quality_campus    int,
    PRIMARY KEY (name, "Date")
);
