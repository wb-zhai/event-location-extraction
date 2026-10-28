BEGIN;

CREATE SCHEMA IF NOT EXISTS article_event_extraction;

-- ---------------------------------------------------------------------------
-- One row per extraction run / model version.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_event_extraction.event_runs (
    run_id                       text PRIMARY KEY,
    ontology_version             text NOT NULL,
    model_name                   text NOT NULL,

    expected_shards              smallint,
    expected_event_articles      bigint,
    expected_events              bigint,
    expected_location_components bigint,

    metadata                     jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at                   timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Shard-level ingestion ledger.
-- Only a successfully committed shard gets a row here.
-- This is the resumability / idempotency control plane.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_event_extraction.event_ingest_shards (
    run_id               text NOT NULL
                         REFERENCES article_event_extraction.event_runs(run_id),

    shard                 smallint NOT NULL
                         CHECK (shard >= 0 AND shard < 200),

    article_rows          bigint NOT NULL,
    event_rows            bigint NOT NULL,
    location_rows         bigint NOT NULL,

    geo_source_sha256     text,
    metadata_sha256       text,

    loaded_at             timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (run_id, shard)
);

-- ---------------------------------------------------------------------------
-- Native V4 events.
--
-- Deliberately NOT FK'd to the legacy risk-factor ontology.
-- event_type is the exact V4 ontology label.
--
-- Large-table indexes are deliberately deferred until after bulk COPY.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_event_extraction.events (
    run_id                         text NOT NULL,

    -- Partitioning / source identity
    publish_month                  date NOT NULL,
    article_publish_date           date NOT NULL,
    shard                           smallint NOT NULL
                                   CHECK (shard >= 0 AND shard < 200),

    -- Stable event identity
    event_key                       bytea NOT NULL
                                   CHECK (octet_length(event_key) = 32),
    article_uri                     text NOT NULL,
    original_event_index            integer NOT NULL
                                   CHECK (original_event_index >= 0),

    -- Article metadata
    language                        text NOT NULL,

    -- Native V4 extraction
    event_type                      text NOT NULL,
    grounding_quote                 text,
    event_location                  text,
    event_location_admin_level      text,
    event_time                      text,
    time_status                     text,
    severity                        text,

    -- Present only where normalization preserved a different original value.
    event_location_raw              text,
    event_location_admin_level_raw  text,

    -- Temporal postprocessing
    raw_event_time                   text,
    temporal_syntax_kind             text,
    temporal_parse_status            text,
    temporal_review_reason           text,
    temporal_start_token             text,
    temporal_end_token               text,
    temporal_start_precision         text,
    temporal_end_precision           text,
    temporal_lower_bound_date        date,
    temporal_upper_bound_date        date,

    -- Safety valve for any future/rare fields not represented above.
    extra                            jsonb

) PARTITION BY RANGE (publish_month);

-- ---------------------------------------------------------------------------
-- One row per geocoded event-location component.
--
-- This preserves multi-component locations rather than collapsing an event
-- to a single article-level adm_code.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_event_extraction.event_locations (
    run_id                         text NOT NULL,

    publish_month                  date NOT NULL,
    shard                          smallint NOT NULL
                                   CHECK (shard >= 0 AND shard < 200),

    event_key                      bytea NOT NULL
                                   CHECK (octet_length(event_key) = 32),

    article_uri                    text NOT NULL,

    component_index                integer NOT NULL
                                   CHECK (component_index >= 0),

    -- Original query/component
    query                          text,
    raw_admin_level                text,

    -- Photon resolution
    photon_layer                   text,
    photon_resolution_status       text,
    photon_selected_query          text,
    photon_selected_layer          text,
    photon_fallback_step           integer,
    photon_fallback_label          text,

    resolved_name                  text,
    zhai                           text,
    photon_type                    text,

    lat                            double precision,
    lon                            double precision,

    osm_id                         text,
    osm_type                       text,

    country                        text,
    countrycode                    text,

    -- World Bank / ZHAI taxonomy
    wb_level                       integer,
    wb_match_status                text,
    adm_code                       text,

    -- Preserve any uncommon materialized geocoding fields.
    extra                          jsonb,

    CHECK (lat IS NULL OR (lat >= -90 AND lat <= 90)),
    CHECK (lon IS NULL OR (lon >= -180 AND lon <= 180))
) PARTITION BY RANGE (publish_month);

COMMENT ON TABLE article_event_extraction.events IS
'Native Global V4 event mentions. event_type retains the extraction ontology label and is not coerced into legacy article-level risk-factor IDs.';

COMMENT ON TABLE article_event_extraction.event_locations IS
'Per-event location components with Photon resolution and World Bank/ZHAI geotaxonomy results.';

COMMIT;
