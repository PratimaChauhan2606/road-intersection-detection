# Business / Impact slide (one slide)

Fill the [brackets] ONLY with numbers from your own results files. Do not use a number you did not measure.

---

## Slide title
**From satellite pixels to a routable, geolocated road database**

## Left column: Problem -> Solution
- **Problem:** road and junction inventories are made by manual surveys: slow, costly, and out of date in fast-growing areas.
- **Solution:** an automated pipeline that turns satellite imagery into a road network graph and a GeoJSON junction inventory that GIS and ITS tools can open directly.
- **Proof on our study area (Gandhinagar, [X] km2):** [X] km of road mapped, [X] intersections found, [P/R/F1] against OpenStreetMap within [15] m.

## Middle column: Who needs this
| User | What they do with it |
|---|---|
| City / smart-city / PWD planning cells | Keep junction and road inventory current without a full survey |
| Signal-coordination and traffic consultants | Start from a geolocated list of junctions and their type (3-way, 4-way, complex) |
| Logistics and fleet-routing teams | Find roads missing from maps in new peri-urban areas; routable graph output |
| National / state multimodal planning (e.g. PM Gati Shakti style GIS layers) | Fast, repeatable road-layer refresh from imagery |

## Right column: How it could earn
- **Mapping-as-a-service:** price per km2 of area mapped.
- **Change-detection subscription:** re-run on new imagery every quarter / year and report new or missing roads.
- **API:** junction and road-graph layer for planning software.
- *(No prices on the slide unless you have a source for them.)*

## Bottom strip: Why our approach
1. Outputs a **graph** (roads + connectivity + junction degree), not just a picture.
2. Built on open-source tools; runs on a single laptop GPU.
3. **Validated** against OpenStreetMap with an explicit distance-tolerance rule, and honest about OSM's limits.

## Honest footer (keep this on the slide - examiners respect it)
*Research prototype. Accuracy measured on one study area. Before real use it needs local validation, and commercial use needs imagery and training data with commercial licences (DeepGlobe is research-licensed; Esri imagery has its own terms).*

---

## What to say if asked "so where is the business?"
"The value is a cheaper, faster, repeatable way to keep road and junction data current. Our prototype shows the technical pipeline works end to end; the next step for a real product is a pilot with one city body, local validation, and commercially licensed imagery."

## What NOT to claim
- Do not say "real-time", "production-ready" or "accurate everywhere".
- Do not quote market sizes or prices without a source.
- Do not say OSM is ground truth.
- Do not compare YOLO mAP with the graph F1 as if they were the same metric.
