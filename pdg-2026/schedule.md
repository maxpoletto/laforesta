# Forest Harvest Scheduling Algorithm

## Input

  - Parcels: list of parcels, each with harvest amount h[i] (from Prelievo (m³))
  - Historical harvests: list of (year, parcel) where year ∈ [-10, -1]
  - Plan horizon: years 0..14 (15 years)

##  Output

  - Schedule: for each parcel, either:
    - (SINGLE, y) — harvest full h[i] in year y
    - (SPLIT, y₁, y₂) — harvest h[i]/2 in years y₁ and y₂
    - (NONE) — not harvested (if h[i] = 0 or unavailable)

## Constraints

  1. 10-year gap: any two harvests of the same parcel must be ≥10 years apart
  2. At most twice: each parcel harvested at most twice in the new plan
  3. Split pairs: if split, must use (y, y+10) where y ∈ [0, 4]

## Objective

  Minimize variance of annual totals: Var(A[0], A[1], ..., A[14])

  where A[y] = Σ contributions to year y

## Algorithm

### Phase 1: Preprocessing

  ```python
  function PREPROCESS(parcels, historical_harvests):
      # Compute earliest available year for each parcel
      for each parcel i:
          past_years = [y for (y, p) in historical_harvests if p == i]
          if past_years is empty:
              earliest[i] = 0
          else:
              earliest[i] = max(0, max(past_years) + 10)

      # Partition parcels by availability
      unavailable = [i for i in parcels if h[i] == 0 or earliest[i] > 14]
      single_only = [i for i in parcels if i not in unavailable and earliest[i] > 4]
      splittable  = [i for i in parcels if i not in unavailable and earliest[i] <= 4]

      # Compute target annual production
      total = Σ h[i] for i not in unavailable
      target = total / 15

      return earliest, unavailable, single_only, splittable, target
  ```

### Phase 2: Greedy Initial Assignment

  ```python
  function GREEDY_ASSIGN(parcels, earliest, target):
      A[0..14] = 0  # annual totals
      schedule = {}

      # Process largest parcels first
      for each parcel i in parcels sorted by h[i] descending:
          if i in unavailable:
              schedule[i] = NONE
              continue

          # Find best single-year assignment
          valid_years = [y for y in 0..14 if y >= earliest[i]]
          best_single = argmin over y in valid_years of |A[y] + h[i] - target|
          single_cost = variance_if_assign(A, i, SINGLE, best_single)

          # Find best split assignment (if splittable)
          if earliest[i] <= 4:
              valid_splits = [(y, y+10) for y in earliest[i]..4]
              best_split = argmin over (y1,y2) in valid_splits of
                           max(|A[y1] + h[i]/2 - target|, |A[y2] + h[i]/2 - target|)
              split_cost = variance_if_assign(A, i, SPLIT, best_split)
          else:
              split_cost = ∞

          # Choose better option
          if split_cost < single_cost:
              schedule[i] = (SPLIT, best_split[0], best_split[1])
              A[best_split[0]] += h[i] / 2
              A[best_split[1]] += h[i] / 2
          else:
              schedule[i] = (SINGLE, best_single)
              A[best_single] += h[i]

      return schedule, A
```

  ### Phase 3: Local Search Optimization

  ```python
  function LOCAL_SEARCH(schedule, A, earliest, max_iterations):
      current_var = variance(A)

      for iter in 1..max_iterations:
          improved = false

          # Try MOVE: change year(s) for one parcel
          for each parcel i with schedule[i] != NONE:
              for each valid alternative assignment alt:
                  new_A = compute_A_with_change(A, schedule, i, alt)
                  if variance(new_A) < current_var:
                      apply_change(schedule, A, i, alt)
                      current_var = variance(new_A)
                      improved = true
                      break

          # Try TOGGLE: switch between single and split
          for each parcel i in splittable:
              if schedule[i] is SINGLE:
                  for each valid split (y1, y2):
                      # try converting to split
                      ...
              else if schedule[i] is SPLIT:
                  for each valid single year y:
                      # try converting to single
                      ...

          # Try SWAP: exchange assignments between two parcels
          for each pair (i, j) of scheduled parcels:
              if can_swap(i, j, earliest):
                  # try swapping their year assignments
                  ...

          if not improved:
              break

      return schedule, A
  ```

  ### Helper Functions

  ```python
  function VARIANCE(A):
      mean = sum(A) / 15
      return Σ (A[y] - mean)² for y in 0..14

  function CAN_ASSIGN(parcel, year, earliest):
      return year >= earliest[parcel]

  function VALID_SINGLE_YEARS(parcel, earliest):
      return [y for y in 0..14 if y >= earliest[parcel]]

  function VALID_SPLIT_PAIRS(parcel, earliest):
      return [(y, y+10) for y in max(0, earliest[parcel])..4]
```

### Main

```python
  function SCHEDULE_HARVESTS(parcels, historical_harvests):
      earliest, unavailable, single_only, splittable, target =
          PREPROCESS(parcels, historical_harvests)

      active_parcels = single_only ∪ splittable

      schedule, A = GREEDY_ASSIGN(active_parcels, earliest, target)

      schedule, A = LOCAL_SEARCH(schedule, A, earliest, max_iterations=1000)

      return schedule, A, variance(A)
```

##  Complexity Notes

  - Parcels: ~50 active
  - Split pairs: only 5 options per parcel
  - Single years: up to 15 options per parcel
  - Local search: polynomial in practice, converges quickly
