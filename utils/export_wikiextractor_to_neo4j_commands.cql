// Create all nodes
USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM "file:///wikisearch_entries.csv" AS row FIELDTERMINATOR "\t"
MERGE (entry:Entry {id: toInt(row.id)})
ON CREATE SET entry.title = row.title, entry.url = row.url, entry.full_text = row.text;

// Create all relationships
USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM "file:///wikisearch_links.csv" AS row FIELDTERMINATOR "\t"
MATCH (source:Entry {title: row.source})
MATCH (target:Entry {title: row.target})
MERGE (source)-[:LINKS_TO]->(target);