# HEAs python scripts

Python scripts to produce supercells of high entropy alloys for DFT simulations.

## mcmaxent.py
The algorithm produces an initial supercell according to the input data (chemical formula, lattice and supercell size). 
This initial supercell is randomily shuffled to maximize the global free volume of the elements by keeping, at the same time,
an even distribution of free volumes.

## qbfmaxent.py
The algorithm produces an initial supercell according to the input data (chemical formula, lattice and supercell size). 
This initial supercell is randomily shuffled and after that the atomic positions are swapped until the global free volume
of the elements is maximazed by keeping, at the same time, an even distribution of free volumes. The algorithm employs a
brute force swapping, i.e., for a given element, all its positions are swapped with all other remaining elements. Many scans,
i.e., swappings all over a given element positions are made until the global free volume remains unchanged near the evenly
distribution of maximum free volume. In order to make the process much less time consuming the "Jar of rocks, pebbles and sand"
principle is followed, i.e., larger free volumes go first. Firstly, the minor element in content has its free volume maximized
by swaping its positions with all others, once it is done its positions keep unchanged. The same principle is applied to the
second minor element in content, now swapping positions with all other remaining elements. This procedure goes on until the
second major element in content. The major element in content just fills the remaining positions.

## qFfTmaxent.py
The algorithm produces an initial supercell according to the input data (chemical formula, lattice and supercell size). 
The supercell is filled up with atoms according to the elemental amounts given in the input formula. The script employs a greedy algorithm
to find the Farthest-first traversal (FfT) walking sequence of the remaining positions up to the atomic amount of the first element, producing,
as much as possible, an even spatial distribution of such atoms on the lattice points. Thus maximazing the free volume as well as reducing the
standard deviation of the mean free volume. The same applies for the standard deviation of the mean volume of the Vornoi cells. The remaining 
elements follow the same procedure, now starting with the remaing lattice positions in the supercell. The element's sequence is chosen according
to the "Jar of rocks, pebbles and sand" principle, i.e., larger free volumes go first. Therefore, the positions of the minor element in content
are firstly optimized followed by the second and so on. The major element in content just fills the last remaining lattice positions.

## qvoromaxent
The algorithm produces an initial supercell according to the input data (lattice and supercell size). 
The supercell is filled up with atoms according to the elemental amounts given in the input formula. The algorithm employs a Voronoi
relaxation (Lloyd's algorithm;  Lloyd, Stuart P. (1982), IEEE Transactions on Information Theory, 28 (2): 129–137) of atomic positions, 
regarding each element, producing, as much as possible, an even distribution of Voronoi cells. It starts from randomic lattice positions of the
first element. A Voronoi tessellation is performed for such positions. The Voronoi cell's centroids are now the new positions and a new Voronoi
tessellation is generated producing new centroids. This process continues until centroids remain nearly the same. The nearst supercell's lattice
positions, regarding such final centroids, are thus filled with atoms of the element. The remaining elements follow the same procedure, now with
the remaing lattice positions in the supercell. The element's sequence is chosen according to the "Jar of rocks, pebbles and sand" principle, i.e.,
larger free volumes go first. Therefore, the positions of the minor element in content are firstly optimized followed by the second, and so on. 
The major element in content just fills the last remaining positions.
