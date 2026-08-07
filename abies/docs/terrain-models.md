# Objective

We want to use freely-available high-resolution (1 m x 1 m) LiDAR data to gain a
better understanding of terrain and vegetation.

Two specific goals:

1. Use Digital Surface Models (DSM) and Digital Terrain Models (DTM) together to
   estimate canopy height at each square-meter patch in the forest. Then build a
   model trained on the  wood volume of sample areas and the LiDAR-measured
   height of those same areas to infer the wood volume of the entire forest from
   LiDAR height data.

2. Use DTM to build high-precision terrain visualizations (maybe using a
   platform like Deck.gl) and, given a set of locations to be harvested, to
   optimize harvest operations (whether to haul from above or below, where to
   build new tracks, etc.)

# Datasets

* DSM FIRST 1x1 Calabria: https://data.europa.eu/data/datasets/m_amte-299fn3-1c04aa0a-5c79-466c-a266-b15428703449

* DSM LAST 1x1 Calabria: https://data.europa.eu/data/datasets/m_amte-299fn3-aa01d612-8741-4340-e7ef-4b7a8d5a07ec

* DTM 1x1 Calabria: https://data.europa.eu/data/datasets/m_amte-299fn3-2cc40856-4a7e-4c84-e96b-02c2d54b674e

# Steps

Here is a breakdown of possible initial steps:

1. Download and clean data, store in canonical but not version-controlled
   location. Note its freshness, coordinate system, and any other important
   metadata.

2. In sequence, expose 3 new "Caratteristiche" layers in Bosco:

   1. an altitude layer that displays terrain altitude as a color map at 1x1
      meter resolution;
   2. a slope layer that is essentially the first derivative of (1), to
      highlight ravines and other high-slope features;
   3. a plant-height layer that displays tree height (presumably as (DSM FIRST -
      DTM)).

3. Built a model to estimate forest volume:

   1. For each sample area in a survey (in this case, specifically, the Sabatino
      survey), compute the aggregate wood volume of measured trees.
   2. Compute the canopy volume as the aggregate volume of 1x1xh rectangular
      parallelepipeds whose heigh is obtained from LiDAR data (DSM FIRST - DTM).
   3. Train a linear model to predict wood volume based on canopy volume.
      (Partition sample areas into test / eval groups.)
   4. Predict whole-forest volume, compare results to harvest plan.

