# Blob-sim-v1
This project is a genetic algorithm based ML to study the behavior of a group of Blobs over generations.


Each Blob starts with a random DNA sample, and tries to get food each day.
The ones with more food have a greater chance at passing their genes to the next genarations.


The behaviors are:
  - Producing (produces food each day when the gene is high)
  - Stealing (steals more when the gene is high. steals only when the gene is above 0.5, otherwise it will produce)
  - Helpers (will have a chance to give food to other blobs with the same gene)
