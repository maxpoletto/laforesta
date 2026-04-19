# Coppice scheduling

For coppice management we are concerned not with volumes, but simply with a schedule. The algorithm does not need tree-level data or growth simulation.

The parcels in scope are those with Governo=Ceduo in bosco/data/particelle.csv.

## Rules

1. A parcel can be harvested with a frequency no greater than that given by the corresponding "Parametro" value in particelle.csv (standard values are 12, 15, 18, and 25 years). The parametro gap is measured from first sub-harvest of one cycle to first sub-harvest of the next.
2. If a parcel exceeds 10 ha in surface area, the harvest must be divided into 10-ha sub-harvests that occur at least two years apart.
3. If two parcels are adjacent (as per columns A and B of bosco/data/cedui-adiacenti.csv), then they cannot be scheduled on the same year, and must be harvested at least two years apart. This applies to all sub-harvest years, not just the first. Adjacency is pairwise, not transitive (e.g. if A-B and B-C are adjacent, A and C are not considered adjacent).

## Scheduling algorithm

The algorithm uses a priority queue:

1. For each ceduo parcel, compute first eligible year = max(anno_inizio, last_harvest + parametro).
2. Insert all parcels into a priority queue sorted by (eligible year, compresa, particella).
3. Pop the next parcel from the queue.
4. Schedule one cycle of sub-harvests:
   - Divide the area into chunks of max 10 ha.
   - For each chunk, find the earliest year that is (a) >= current eligible year, (b) >= previous sub-harvest year + 2 (for 2nd+ chunks), and (c) has no adjacency conflict with already-scheduled events.
5. Compute next cycle eligible year = first sub-harvest year of this cycle + parametro. If within the planning range, re-insert the parcel into the queue.
6. Repeat until the queue is empty.

Past harvest dates come from calendario-mannesi.csv. Parcels with no harvest history get last_harvest = 0, making them eligible immediately. To break ties in the priority queue, non-adjacent parcels in the same compresa are chosen in lexicographic order, then parcels in different comprese.

## Examples

* A standalone parcel of 8 ha with parametro=12 was last harvested in 2015. It can be harvested in 2027 and then again in 2039.

* A standalone parcel of 25 ha with parametro=12 was last harvested in 2015. It is harvested in 2027 (10 ha), 2029 (10 ha), and 2031 (5 ha), and then similarly in 2039, 2041, and 2043.

* Two adjacent parcels of 8 ha with parametro=12 were last harvested in 2015 (parcel A) and 2016 (B). A is scheduled in 2027 and 2039. B cannot be scheduled in 2028 because that is only one year after 2027, so it is scheduled for 2029 and 2041.

* Two adjacent parcels of 12 ha with parametro=12 were last harvested in 2015 (A) and 2016 (B). A is scheduled in 2027 (10 ha), 2029 (2 ha), 2039, and 2041. B cannot be scheduled in 2028-2030. It is pushed to 2031 and 2033, and then to 2043 and 2045.

## Directive syntax

@@calendario_ceduo(particelle=FILE,calendario=FILE,adiacenze=FILE,anno_inizio=M,anno_fine=N)

Output: a table with columns Anno, Compresa, Particella, Superficie (ha), Note. The "Note" column says "Continuazione intervento YYYY" for second and subsequent sub-harvests where YYYY is the year of the first sub-harvest of that cycle.

@@tabella_ceduo(particelle=FILE,calendario=FILE,adiacenze=FILE,anno_inizio=M,anno_fine=N)

Output: a Gantt-style chart illustrating the lifetime of each batch of 50
preserved shoots ("matricine"). One row per ceduo parcel (sorted in natural
order by compresa then particella), one bar per sub-harvest. A bar starts at its
sub-harvest year and ends `2 * parametro` years later (the batch is kept through
the following harvest and cut at the second one). Parcels larger than 10 ha have
multiple sub-harvest slots and therefore taller rows. Bars extending beyond
`anno_fine` are drawn with a rightward overflow arrow.
