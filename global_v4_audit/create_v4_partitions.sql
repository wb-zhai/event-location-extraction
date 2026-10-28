BEGIN;

DO $$
DECLARE
    m date;
    next_m date;
    event_part text;
    location_part text;
BEGIN
    -- Global V4 observed publication range:
    -- min = 2013-12-18 13:07:00
    -- max = 2026-06-28 05:48:56
    FOR m IN
        SELECT generate_series(
            DATE '2013-12-01',
            DATE '2026-06-01',
            INTERVAL '1 month'
        )::date
    LOOP
        next_m := (m + INTERVAL '1 month')::date;

        event_part :=
            'events_p' || to_char(m, 'YYYYMM');

        location_part :=
            'event_locations_p' || to_char(m, 'YYYYMM');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS article_event_extraction.%I
             PARTITION OF article_event_extraction.events
             FOR VALUES FROM (%L) TO (%L)',
            event_part,
            m,
            next_m
        );

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS article_event_extraction.%I
             PARTITION OF article_event_extraction.event_locations
             FOR VALUES FROM (%L) TO (%L)',
            location_part,
            m,
            next_m
        );
    END LOOP;
END
$$;

COMMIT;
