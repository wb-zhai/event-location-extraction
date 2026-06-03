# Annotation Guidelines

These rules are for both human annotators and LLM annotation prompts. The goal
is a small, consistent set of exact spans that can be verified against the
article text. Prefer precision over recall when the evidence is unclear.

## Source of Truth

Use only:

1. `article_text` for spans, offsets, events, locations, and arguments.
2. The title as context only. Never annotate a span from the title.
3. `ontologies/zhai/ontology.json` for allowed event types, argument roles, and
   location types.

Do not use world knowledge to add missing details. Do not invent labels,
locations, offsets, or paraphrased text.

## Output Fields

Each record contains:

- `events`: ontology event trigger spans. Each event has `event_type`, `start`,
  `end`, `text`, and `arguments`.
- `events[*].arguments`: explicit location mentions linked to that event. Each
  argument has `role`, `start`, `end`, `text`, and `location_type`.
- `locations`: every explicit location mention in `article_text`, whether or
  not it is linked to an event.
- `has_target_event`: `true` when at least one event is annotated, otherwise
  `false`.
- `negative_reason`: a concise reason only when `has_target_event` is `false`;
  otherwise `null`.

## Recommended Annotation Order

1. Read the article text and identify explicit location mentions.
2. Identify event trigger candidates asserted in the article text.
3. Match each event candidate to the most specific ontology event type.
4. Select the shortest natural trigger span for each accepted event.
5. Link only explicitly mentioned location arguments to each event.
6. Validate every `text`, `start`, and `end` against `article_text`.
7. Remove unsupported, duplicate, overlapping, or inferred annotations.

## Event Assertion Rules

Annotate an event only when the article text asserts that the event, condition,
threat, risk, impact, or trend exists, happened, is happening, worsened, is
expected, or directly affects people, assets, systems, or the environment.

Do annotate:

- Current or recent events reported by the article.
- Ongoing conditions, such as shortages, displacement, drought, or insecurity.
- Forecasts, warnings, threats, or risks when the text presents them as real
  concerns.
- Article-relevant background events when the text uses them to explain the
  current situation.

Do not annotate:

- Topic-only mentions with no asserted event or condition.
- Figurative language, such as `a flood of criticism`.
- Purely hypothetical examples not tied to a real risk, warning, or forecast.
- Historical background that is not relevant to the article's reported
  situation.
- Events that appear only in the title and not in `article_text`.

## Event Trigger Spans

Annotate the shortest natural phrase that directly expresses the event. The
span should be readable as the event mention, but should not include arguments,
causes, dates, casualty counts, or locations unless those words are required to
name the event type.

Good trigger spans:

- `flooding`
- `food shortages`
- `power outage`
- `left their homes`
- `risk of attacks`
- `monsoon rains`
- `desert locust outbreak`
- `record rains`
- `food insecurity`
- `humanitarian aid`
- `sanctions`
- `fired hundreds of rockets`
- `missiles are incoming`
- `flash floods`
- `scarce and increasingly stressed water resources`
- `prices have continued to increase`
- `Emergency (IPC Phase 4) outcomes are expected`
- `yields for key cereals, such as sorghum and millet, are lower than usual`
- `barriers to humanitarian aid`
- `donated rice`
- `cancelled 2,155 flights`

Avoid overlong spans:

- Prefer `strikes`, not `168 strikes last week in Iraq and Syria`.
- Prefer `flooding`, not `flooding across several northern districts`.
- Prefer `food shortages`, not `severe food shortages in northern camps`.

Avoid under-specific spans:

- Prefer `air raids`, not `raids`, when `air` changes the event.
- Prefer `monsoon rains`, not `rains`, for monsoon-specific rainfall events.
- Prefer `power outage`, not `outage`, when the full phrase names the event.
- Prefer `fired hundreds of rockets`, not `fired`, when the object makes the
  event identifiable.
- Prefer `prices have continued to increase`, not `increase`, when the phrase
  expresses the economic event.
- Prefer `barriers to humanitarian aid`, not `barriers`, when the object
  determines the event type.

If a sentence contains several distinct ontology events, annotate each distinct
trigger. If repeated mentions refer to the same event in different places in
the article, annotate each explicit trigger span unless it is a duplicate at
the same offsets.

## Event Type Selection

Choose the most specific ontology event type supported by the trigger phrase
and its sentence context.

If two labels are plausible:

- Use the label whose ontology definition best matches the text.
- Prefer concrete event labels over generic risk labels.
- Use a generic risk label only when the article describes risk without a more
  specific asserted event.
- Omit the event when no ontology label is genuinely supported.

Examples:

- `heavy rains caused flooding`: annotate `flooding`; annotate `increased
  rainfall` only if increased rainfall itself is the reported event.
- `risk of attacks`: annotate `threat of attack`, not a broader violence label,
  if the ontology definition supports that label.
- `armed clashes`: annotate `military conflict` unless the article only reports
  general insecurity without armed confrontation.
- `inflation has made food unaffordable`: annotate an economic people-impact
  event only if the ontology definition supports the described impact.
- `record rains` and `flooding`: annotate both when the article reports the
  rainfall event and the resulting flood impact as distinct events.
- `humanitarian aid`: annotate a food-aid or financial-contribution event only
  when the sentence says aid is needed, provided, funded, blocked, or targeted.
- `sanctions`: annotate `trade embargo` when the text describes sanctions as
  restrictions affecting trade, aid, finance, or access.

## Reviewed Calibration Examples

The following examples are drawn from reviewed manual fixes. Use them as
calibration examples, not as a closed list of allowed triggers.

Weather, hazards, and infrastructure:

- `record rains` -> `increased rainfall`; linked location:
  `United Arab Emirates` as `country`.
- `flooding` -> `flooding`; linked location: `United Arab Emirates` as
  `country`.
- `flooded` -> `flooding`; linked location: `Dubai` as `city`.
- `hobbling Dubai airport` -> `flight disruption`; linked location: `Dubai` as
  `city`.
- `water-clogged roads` -> `infrastructure degradation`; linked location:
  `Dubai` as `city`.
- `Cholera` -> `disease outbreak`; linked locations: `Narok` as `county` and
  `Nairobi` as `city`.

Food security, agriculture, and water:

- `desert locust outbreak` -> `pest insect spread`; linked location:
  `Horn of Africa` as `other`.
- `infestation` -> `pest insect spread`; linked location: `Kenya` as
  `country`.
- `food insecurity` -> `food scarcity`; linked location: `Kenya` as `country`.
- `Emergency (IPC Phase 4) outcomes are expected` -> `food scarcity`; linked
  location: `South Kordofan` as `state`.
- `yields for key cereals, such as sorghum and millet, are lower than usual`
  -> `low crop yield`; linked location: `Sudan` as `country`.
- `Shrinking water sources` -> `water supply reduction`; omit event arguments
  if the sentence does not explicitly link a location.

Conflict, violence, and access:

- `blast` -> `violence`; linked locations: `Pakistan` as `country`,
  `Balochistan` as `state`, and `Chaman` as `city`.
- `strikes` -> `military conflict`; linked locations may include both the
  country and a more specific city when both are explicit, such as `Syria` and
  `Hajin`.
- `Shelling` -> `artillery bombing`; linked location: `Hodeida` as `city`.
- `blockade` -> `corridor blockade` when the text describes a route, port, or
  access corridor being blocked.
- `barriers to humanitarian aid` -> `obstruction of humanitarian aid`.

Aid, sanctions, and economic conditions:

- `humanitarian support` -> `financial contributions`; linked arguments:
  `EU` as `source_location` and `Kenya` as `target_location`.
- `humanitarian aid` -> `food aid requirement`; linked argument:
  `North Korea` as `target_location` when the aid is directed there.
- `donated rice` -> `food aid requirement`; linked target locations can be a
  list of explicit destination cities, such as `Maiduguri`, `Damaturu`, `Yola`,
  `Jalingo`, `Gombe`, and `Bauchi`.
- `sanctions` -> `trade embargo`; linked location: `North Korea` as `country`
  when the text says the sanctions apply to North Korea.
- `prices have continued to increase` -> `economic situation affecting people`;
  linked location: `Sudan` as `country`.
- `continued depreciation of the Sudanese Pound` -> `economic situation
  affecting people`; linked location: `Sudan` as `country`.

## Location Mentions

Extract every explicit place mention in `article_text`.

Use ontology location types as follows:

- `country`: sovereign countries or nationally scoped place mentions.
- `state`: first-order administrative divisions, such as states, provinces, or
  governorates.
- `county`: second-order administrative divisions, such as districts,
  municipalities, prefectures, or counties.
- `city`: cities, towns, and villages.
- `other`: camps, border areas, regions, neighborhoods, facilities, rivers,
  coastal zones, roads, ports, airports, mountains, deserts, and other
  place-like mentions.

Do include:

- Named places, such as `Sudan`, `Gaza`, `Lagos`, or `Kakuma camp`.
- Common-noun place mentions when used as places in context, such as `the
  capital`, `the border`, `the river`, or `nearby villages`.
- Repeated location mentions at different offsets.

Do not include:

- Demonyms or nationality adjectives unless the text uses them as a place
  mention.
- Organizations that merely contain place names, unless the place itself is
  separately mentioned.
- Person names, government bodies, armed groups, companies, or agencies as
  locations.
- Implied locations not present in the text.

Examples:

- In `Pakistani officials visited Islamabad`, annotate `Islamabad`; do not
  annotate `Pakistani`.
- In `the Sudanese army entered villages near the border`, annotate `villages`
  and `border` if they are place mentions in context; do not annotate
  `Sudanese` as a location.
- If the text says `the capital` without naming the city, annotate `capital` as
  `other` only when it is used as a place mention.

## Location Arguments

An event argument is a location mention explicitly linked to a specific event.
Arguments must also appear in the top-level `locations` list.

Use only roles allowed by `event_argument_roles` for the selected event type:

- `location`: where the event occurs, exists, is observed, or affects people,
  livelihoods, markets, crops, livestock, resources, infrastructure, or systems.
- `source_location`: where movement, action, pressure, hazard, people, goods,
  or aid originates.
- `target_location`: where movement, action, pressure, hazard, people, goods,
  or aid is directed to or ends.

Link a location to an event when the text explicitly connects them in the same
sentence or nearby context. Do not link a location just because it appears in
the article. When the linkage is unclear, keep the location in `locations` but
omit it from the event's `arguments`.

Examples:

- `humanitarian support` from `EU` for `Kenya`: `EU` is `source_location`;
  `Kenya` is `target_location`.
- `humanitarian aid` for `North Korea`: `North Korea` is `target_location`
  when the aid is directed there.
- `donated rice` sent to `Maiduguri`, `Damaturu`, `Yola`, `Jalingo`, `Gombe`,
  and `Bauchi`: each destination city is a `target_location`.
- `record rains` in the `United Arab Emirates`: `United Arab Emirates` is
  `location`.
- `strikes` near `Syria's Hajin`: link both `Syria` and `Hajin` as
  `location` when the text explicitly names both the broader country and the
  specific place.
- `blast at mosque in southwest Pakistan`: link `Pakistan`, `Balochistan`, and
  `Chaman` only if each place is explicitly present in the article text.

## Span and Offset Rules

Every span must be an exact substring of `article_text`.

Rules:

- Offsets are zero-based character offsets into `article_text`.
- `end` is exclusive.
- `text` must equal `article_text[start:end]` exactly.
- Preserve original casing, accents, punctuation, and whitespace inside the
  selected span.
- Exclude surrounding quotes, parentheses, commas, periods, and other trailing
  punctuation unless they are part of the name.
- Exclude leading articles such as `the`, `a`, and `an` unless they are part of
  a proper name or needed for a common-noun place mention.
- Do not trim internal punctuation from names such as `Cote d'Ivoire` or
  `Borno-State` if that punctuation appears in the text.
- Never create approximate offsets. If exact offsets cannot be verified, omit
  the annotation.

For LLMs: before returning JSON, check each span by comparing `text` to the
substring at `start:end`.

For humans: when correcting a span, correct both the text and offsets; do not
leave either field stale.

## Overlaps and Duplicates

Overlapping top-level event and location spans are not accepted for token
classification. Resolve overlaps before finalizing the record.

Rules:

- Prefer the shortest valid event trigger that excludes linked locations.
- Prefer a location span over a broader event span when the broader event span
  wrongly includes the location.
- Do not annotate nested location spans unless the shorter and longer spans are
  separate explicit place mentions needed for review. If nested spans create a
  token overlap, keep the most specific useful span.
- Deduplicate annotations with the same label, `start`, and `end`.
- Allow the same location span to be linked to multiple events when the text
  explicitly supports each link.

## Negative Records

A negative record means no ontology event is asserted in the article text or
the current annotation window.

For negative records:

- Set `events` to an empty list.
- Set `has_target_event` to `false`.
- Provide a concise `negative_reason`.
- Still extract all explicit locations.

Common hard negatives:

- Opinion or analysis that discusses a domain, actor, or policy but asserts no
  ontology event.
- Articles mentioning countries and military actors but no attack, threat,
  displacement, casualty, humanitarian condition, or other ontology event.
- Economic reporting that mentions prices, markets, or forecasts but not an
  ontology economic decline or people-impacting economic situation.
- Articles where the only target event appears in the title, not in
  `article_text`.

## Ambiguity Policy

When uncertain:

- Annotate only what is explicitly supported by `article_text`.
- Prefer omitting a weak event over guessing.
- Prefer keeping an unlinked location in `locations` over inventing an event
  argument link.
- Use `other` for explicit place mentions that do not cleanly fit country,
  state, county, or city.
- Never paraphrase span text.
- Never output reasoning, comments, Markdown, or explanatory text in the JSON
  response.
