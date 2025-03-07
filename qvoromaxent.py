#!/usr/bin/env python

# Quick Voronoi Maximum Entropy (qvoromaxent)
# august 2022

# ASE (Atomic Simulation Environment) python script to
# quickly produce supercells of bcc,fcc or hcp lattices according to the
# maximum entropy principle (maximum free volume for each element).
# Refs:
# S.Q. Wang, Entropy, 15 (2013), p. 5536
# A.D. Wissner-Gross, C.E. Freer, Phys. Rev. Lett., 110 (2013), p. 168702
# S.-M. Zheng, W.-Q. Feng, S.-Q. Wang, Comp. Mat. Sci., 142 (2018), p. 332
#
# The algorithm produces an initial supercell according to the input data (lattice and supercell size). 
# The supercell is filled up with atoms according to the elemental amounts given in the input formula. The algorithm employs a Voronoi
# relaxation (Lloyd's algorithm;  Lloyd, Stuart P. (1982), IEEE Transactions on Information Theory, 28 (2): 129–137) of atomic positions, 
# regarding each element, producing, as much as possible, an even distribution of Voronoi cells. It starts from randomic lattice positions of the
# first element. A Voronoi tessellation is performed for such positions. The Voronoi cell's centroids are now the new positions and a new Voronoi
# tessellation is generated producing new centroids. This process continues until centroids remain nearly the same. The nearst supercell's lattice
# positions, regarding such final centroids, are thus filled with atoms of the element. The remaining elements follow the same procedure, now with
# the remaing lattice positions in the supercell. The element's sequence is chosen according to the "Jar of rocks, pebbles and sand" principle, i.e.,
# larger free volumes go first. Therefore, the positions of the minor element in content are firstly optimized followed by the second, and so on. 
# The major element in content just fills the last remaining positions.

# HOW TO CITE  
# If you use this script in your work, please cite:  
# J.H. Mazo, C. Soares, G.K. Inui, M.F. de Oliveira, J.L.F. Da Silva, Materials Science and Engineering: A, 929 (2025), p. 148053  
# https://doi.org/10.1016/j.msea.2025.148053  

# Python packages needed:
# ase: Atomic Simulation Environment (ASE) https://gitlab.com/ase/ase
# spglib: Software library for crystal symmetry search (Spglib) https://atztogo.github.io/spglib/
# 
# Both packages are also avaliable in Anaconda environment: 
# https://anaconda.org/conda-forge/ase
# https://anaconda.org/conda-forge/spglib
# 
#  How to cite
#------------------------------------------------------------
#  Authors, Paper Title, Journal Name, Year. (TO BE DEFINED)
#------------------------------------------------------------  

howtouse = '''
# Command Line
#--------------------------
# $ python qvoromaxent.py argv1 argv2 argv3 argv4 argv5
#
# argv1 = chemical formula in Hill notation
# argv2 = base lattice (bcc, fcc or hcp)
# argv3, argv4, argv5 = natural numbers, repetitions of the base cell in x, y and z to produce a supercell
# argv6 = optional verbose, if you want verbose output
# argv7 = optional view, if you want ASE-GUI for crystal visualization
#
'''
# Outputs:
# - verbose running status on the fly
# - final structure(spacegroup), supercell parameters, atomic positions and statistics at the end
# - files:
#   argv1_argv2argv3argv4argv5.txt
#		text file with the final results, structure(spacegroup), supercell parameters, statistics and atomic positions (appended file)
# 	argv1_argv2argv3argv4argv5_spacegroup_final.cif
#		cif file with the supercell final structure
#

import sys
import numpy as np
import random as rnd
import spglib
import re
from datetime import datetime
from ase.spacegroup import crystal
from ase.visualize import view
from ase.formula import Formula
from ase import io
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree

# Function to find the distances of all atoms from a given atom regarding the same element
def dists(indice_):
	indices_=[i_ for i_ in range(len(supercelula)) if supercelula[i_].symbol==supercelula[indice_].symbol]
	dists_=supercelula.get_distances(indice_,indices_,mic=True) # mic=True for periodic bondaries
	return dists_

# Function to find the total length, all distances among atoms of the same element
def totalL():
	totalL_=0
	for i_ in range(len(supercelula)):
		totalL_+=sum(dists(i_))
	totalL_=totalL_/2 # since each distance is summed twice
	return totalL_

# Function to find the total free length (minimum distance among atoms) regarding each element
def freeL():
	freeL_=0
	for i_ in range(len(supercelula)):
		freeL_+=sorted(dists(i_))[1] # the first distance is from the atom to itself (0), the second is the nearest neighbour
	return freeL_

# Function to find the total free volume (non overlapping centered spheres regarding each element)
def freeV():
	freeV_=0
	for i_ in range(len(supercelula)):
		freeV_+=4/3*np.pi*(sorted(dists(i_))[1]/2)**3
	return freeV_

# Function to produce periodic boundary condition
def pbc(pontos_):
	mirr_=[i_ for i_ in pontos_]
	for i_ in range(len(mirr_)):
		for x_ in [-supercelula.cell[0],np.array([0,0,0]),supercelula.cell[0]]:
			for y_ in [-supercelula.cell[1],np.array([0,0,0]),supercelula.cell[1]]:
				for z_ in [-supercelula.cell[2],np.array([0,0,0]),supercelula.cell[2]]:
					if not (sum(x_)==0 and sum(y_)==0 and sum(z_)==0):
						mirr_.append(mirr_[i_]+x_+y_+z_)
	return mirr_

# Voronoi volumes of a given element
def vorvol(elemento_):
	pontos_=[supercelula[i_].position for i_ in range(len(supercelula)) if supercelula[i_].symbol==elemento_]
	vor_=Voronoi(pbc(pontos_))
	vols_=np.array([ConvexHull(vor_.vertices[vor_.regions[vor_.point_region[i_]]]).volume for i_ in range(len(pontos_))])
	return vols_

# points centralization in the supercell (not on the grid positions)
def centra(pontos_):
	base_=np.array([supercelula.cell[i_] for i_ in range(3)])
	dif_=np.mean(pontos_,axis=0)-np.mean(base_,axis=0)
	pontos_=pontos_-dif_
	return pontos_

# Voroni optimization of a given set of points
def optvoro(pontos_,precision_):
	while True:
		soma_=0
		pontos_=centra(pontos_) # center points regarding the supercell (not on grid positions)
		vor_=Voronoi(pbc(pontos_))
		for i_ in range(len(pontos_)-1): # é preciso fixar um ponto
			medio_=np.mean(vor_.vertices[vor_.regions[vor_.point_region[i_]]],axis=0)
			soma_+=np.linalg.norm(pontos_[i_]-medio_)
			pontos_[i_]=medio_
		if soma_<precision_: break
	vols_=np.array([ConvexHull(vor_.vertices[vor_.regions[vor_.point_region[i_]]]).volume for i_ in range(len(pontos_))])
	if verbose:
		print('Mean Vol.',str(np.mean(vols_)),'StD.',str(np.std(vols_)))
	return pontos_

# Find nearst grid positions for a given set of generic points
def nearst(grid_,pontos_):
	n_=12 # how much neighbours are searched for every point, 12 usually works fine
	arvore_=KDTree(grid_)
	# dictionary of neighbours
	vizinhos_=dict((i_,arvore_.query(pontos_,k=n_)[1][i_]) for i_ in range(len(pontos_)))
	# dictionary of distances, same keys as the previous
	dists_=dict((i_,arvore_.query(pontos_,k=n_)[0][i_]) for i_ in range(len(pontos_)))
	# reverse sort the dictionary rows according to first distance length (columns are already sorted)
	dists_=dict(sorted(dists_.items(), key=lambda item_: item_[1][0], reverse=True))
	# search for the shortest distances and not repeating neighbours
	lista_=[]
	for i_ in dists_.keys():
		for j_ in range(n_):
			if vizinhos_[i_][j_] not in lista_:
				lista_.append(vizinhos_[i_][j_])
				break
	if len(lista_)!=len(pontos_):
		sys.exit('Increase the number of neighbours to search.')
	pos_=[grid[i_] for i_ in lista_]
	return pos_

# Find supercell indices for a given set of grid points
def findindex(gridpos_):
	malha_=supercelula.get_positions(False).tolist()
	indices_=[malha_.index(gridpos_[i_].tolist()) for i_ in range(len(gridpos_))]
	return indices_


# Main program 

# atomic radii (angstroms)
# from: Miracle, D. B.; Sanders, W. S.; Philosophical Magazine, v. 83, n. 20, p. 2418, 2003.
raios={'Ag':1.42,'Al':1.43,'Au':1.45,'B':0.78,'Be':1.12,'C':0.77,'Ca':1.97,'Ce':1.82,'Co':1.28,'Cr':1.28,'Cu':1.27,'Dy':1.77,'Er':1.76,
	   'Fe':1.28,'Ga':1.32,'Gd':1.74,'Ge':1.14,'Hf':1.67,'La':1.87,'Mg':1.6,'Mn':1.32,'Mo':1.39, 'Nb':1.46,'Nd':1.85,'Ni':1.28,'P':1.0,
	   'Pb':1.75,'Pd':1.41,'Pt':1.38,'Rh':1.34,'Si':1.02,'Sn':1.62,'Ta':1.49,'Th':1.8,'Ti':1.46,'U':1.58,'V':1.34,'Y':1.8,
	   'Zn':1.38,'Zr':1.58}

print('\b')

# Initial time
t_inicio=datetime.now()

# Input data
try:
	compo=Formula(sys.argv[1]).count() # element symbols and atomic amounts from the chemical formula in Hill notation
except:
	sys.exit(howtouse)
gruponumero={'bcc':229,'fcc':225,'hcp':194} # space group numbers
if sys.argv[2] in gruponumero.keys():
	grupoespacial=gruponumero[sys.argv[2]] 
else:
	sys.exit('Base structure must be bcc, fcc or hcp')
try:
	rx=int(sys.argv[3]) # base cell repetitions in x
	ry=int(sys.argv[4]) # base cell repetitions in y
	rz=int(sys.argv[5]) # base cell repetitions in z
except:
	sys.exit('Base cell repetitions must be natural numbers for all directions.')
if rx<1 or ry<1 or rz<1:
	sys.exit('Supercell must be at least 1x1x1')
verbose=False
if 'verbose' in sys.argv:
	verbose=True # activates the verbose output
ver=False
if 'view' in sys.argv:
	ver=True # activates the ase GUI crystal viewer


# Converts the atomic amounts into fractions
compo=dict(zip(compo.keys(),[i/sum(compo.values()) for i in compo.values()]))

# Sort dictionary of elments according to their amounts
compo=dict(sorted(compo.items(), key=lambda item: item[1]))

# Estimative of the base cell parameter, in angstroms,
# calculated from a weighted average of the atomic radii 
rmedio=sum(list(compo.values())*np.array([raios[i] for i in compo.keys()]))
if grupoespacial==229:
	pa=4*rmedio/(3**(1/2)) # base cell parameter for bcc
	pc=pa; aa=90; ab=90; ac=90; base=(0,0,0)
elif grupoespacial==225:
	pa=4*rmedio/(2**(1/2)) # base cell parameter for fcc
	pc=pa; aa=90; ab=90; ac=90; base=(0,0,0)
elif grupoespacial==194:
	pa=2*rmedio # base cell parameter for hcp       
	pc=pa*(8/3)**(1/2) # ideal c/a ratio
	aa=90; ab=90; ac=120; base=(1./3.,2./3.,3./4.)

# Builds the supercell filled up with any element
supercelula=crystal('H',[base],spacegroup=grupoespacial,cellpar=[pa,pa,pc,aa,ab,ac],size=(rx,ry,rz))

# Check if the supercell has at least two atoms of each element
# Check if the chemical formula can be properly reproduced with the amount of atoms in the supercell
totalelemento=[round(i*len(supercelula)) for i in compo.values()]
for i in totalelemento:
	if i<2:
		sys.exit('The chemical formula must produce at least two positions for each element in the supercell, change its size.')
if sum(totalelemento)!=len(supercelula):
	sys.exit('The chemical formula can\'t be properly reproduced in the supercell, change its size.')

# Initial conditions
print('-- qvoromaxent --')
print('Number of atoms: '+str(len(supercelula)))
print('Supercell Chemical Formula: ',end='')
for i in reversed(compo.keys()):
	por = re.sub(r'\.0','',str(round(compo[i]*100,1))) # takes amount as percentage and removes .0
	print(i+por,end='')
print('\n\b')

# Preparing the main loop
resta=[i for i in range(len(supercelula))] # list of remaing positions in the supercell

# Main loop to optimaize the Voronoi tesselations for each element
# Follow the element amounts in order to apply the "Jar of rocks, pebbles and sand" principle
# i.e., larger free volumes are optimized first
els=list(compo.keys())
ultimo=els[len(els)-1]
els.remove(ultimo)
for elemento in els:
	if verbose:
		print(elemento)
	# sort not filled positions in the supercell
	indices=[resta[rnd.randint(0,len(resta))-1]]
	for i in range(round(len(supercelula)*compo[elemento])-1):
		a=indices[0]
		while a in indices: # check if the sorted index has already been sorted 
			a=resta[rnd.randint(0,len(resta)-1)]
		indices.append(a)
	# Get the positions of the sorted indices
	pontos=np.array([supercelula[i].position for i in indices])
	# Voronoi optimization
	if verbose:
		print('Voronoi relaxation')
		print('Initial indices')
		print(indices)
	pontos=optvoro(pontos,1e-2)
	# Find nearst remaining supercell positions
	grid=[supercelula[i].position for i in resta]
	indices=findindex(nearst(grid,pontos))
	if verbose:
		print('Final indices')
		print(indices)
	# Fill the supercell positions with the current element
	# and removes such indices from the remaining list
	for i in indices:
		supercelula[i].symbol=elemento
		resta.remove(i)
# Fills the remaining positions with the last element
for i in resta:
	supercelula[i].symbol=ultimo
if verbose:
	print(ultimo)
	print('Final indices')
	print(resta)

# Main loop end


# Final conditions	
print('\b')
print('Final Total Length: '+str(totalL()))
print('Final Free Length: '+str(freeL()))
print('Final Free Volume: '+str(freeV()))

# Statistics for each element
statistics=[] # lines of the statistics report
elFV=[] # free volumes of each atom regarding each element
Lista={} # index list of each element
for i in compo:
	Lista[i]=[]
for i in range(len(supercelula)):
	Lista[supercelula[i].symbol].append(i)
# Maximum possible free volume for each element (average volume)
# theoretical max free vol. is a fcc(hcp) packing of equal spheres
tavV=dict((i,supercelula.cell.volume/len(Lista[i])*0.74) for i in compo.keys())
statistics.append('Element\t| Theor. Free Vol.\t| Mean Free Vol.\t| Std. Deviation\t| Voronoi Mean Vol.\t| Std. Deviation')
for i in Lista:
	for j in Lista[i]:
		elFV.append(4/3*np.pi*(sorted(dists(j))[1]/2)**3)

	statistics.append(i+"\t| {0:8.12f}\t| {1:8.12f}\t| {2:8.12f}\t| {3:8.12f}\t| {4:8.12f}".format(tavV[i],np.mean(elFV),np.std(elFV),np.mean(vorvol(i)),np.std(vorvol(i))))
	elFV.clear()

# Statistics output
print('\b')
for i in statistics:
	print(i)
print('\b')

# Simmetry verification and space group determination
cell=(supercelula.get_cell(),supercelula.get_scaled_positions(),supercelula.get_atomic_numbers())
grupo=spglib.get_spacegroup(cell, symprec=1e-5)
data_set=spglib.get_symmetry_dataset(cell)
print('Space Group: ',grupo)
print('Rotation matrix', data_set.get('std_rotation_matrix'))

# Gets the Hermann-Mauguin notation and exchanges / by ! in order to use it in file names 
hm=re.sub(r'/',r'!',re.match('^\S+',grupo).group(0))

# Outputs supercell parameters and final atomic positions
print('Supercell')
print(supercelula.cell[0])
print(supercelula.cell[1])
print(supercelula.cell[2])
print('\b')
print('Final Positions')
for i in range(len(supercelula)):
	print(supercelula[i].symbol,'\t',supercelula[i].scaled_position[0],'\t',supercelula[i].scaled_position[1],'\t',supercelula[i].scaled_position[2])
print('\b')

# Outputs the final cif file
io.write(sys.argv[1]+'_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'_'+hm+'_final.cif',supercelula,'cif')

# Outputs the txt file
out = open(sys.argv[1]+'_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'.txt', "a")
out.write('-- qvoromaxent --'+'\n')
out.write(sys.argv[1]+' '+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+' '+str(datetime.now())+'\n')
out.write('Number of atoms:'+str(len(supercelula))+'\n')
out.write('Supercell Chemical Formula: ')
for i in reversed(Lista):
	subs = str(round(compo[i]*100,1))
	subs = re.sub(r'\.0','',subs) # removes .0
	out.write(i+str(subs))
out.write('\n')
out.write('Final conditions:'+'\n')
out.write('  Total length: '+str(totalL())+'\n')
out.write('  Free length: '+str(freeL())+'\n')
out.write('  Free volume: '+str(freeV())+'\n')
# Statistics for each element
out.write('\n')
for i in statistics:
	out.write(i+'\n')
out.write('\n')
out.write('Space group: '+grupo+'\n')
out.write('Supercell\n')
out.write(str(supercelula.cell[0])+'\n')
out.write(str(supercelula.cell[1])+'\n')
out.write(str(supercelula.cell[2])+'\n')
out.write('\n')
out.write('Atomic positions (scaled)\n')
for i in range(len(supercelula)):
	out.write(supercelula[i].symbol+'\t'+str(supercelula[i].scaled_position[0])+'\t'+str(supercelula[i].scaled_position[1])+'\t'+str(supercelula[i].scaled_position[2])+'\n')
out.write('\n')
out.write('Exec. time: {}'.format(datetime.now()-t_inicio))
out.write('\n\n\n')	
out.close()

if ver:
	view(supercelula) # ase GUI output

print('Exec. time: {}'.format(datetime.now()-t_inicio))
print('\b')
