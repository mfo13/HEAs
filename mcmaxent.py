#!/usr/bin/env python

# Monte Carlo Maximum Entropy (mcmaxent)
# july-august 2022

# ASE (Atomic Simulation Environment) python script to
# produce supercells of bcc,fcc or hcp lattices according to the
# maximum entropy principle (elements maximum free volume).
# Refs:
# S.Q. Wang, Entropy, 15 (2013), p. 5536
# A.D. Wissner-Gross, C.E. Freer, Phys. Rev. Lett., 110 (2013), p. 168702
# S.-M. Zheng, W.-Q. Feng, S.-Q. Wang, Comp. Mat. Sci., 142 (2018), p. 332
#
# The algorithm produces an initial supercell according to the input data (chemical formula, lattice and supercell size). 
# This initial supercell is randomily shuffled to maximize the global free volume of the elements by keeping, at the same time,
# an even distribution of free volumes.

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

howtouse = '''
# Command Line
#--------------------------
# $ python mcmaxent.py argv1 argv2 argv3 argv4 argv5 argv6 argv7
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
# 	argv1_argv2argv3argv4argv5_initial.cif
#		cif file with the initial supercell structure
#	argv1_argv2argv3argv4argv5_spacegroup_final.cif
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
	aa=120; ab=90; ac=90; base=(1./3.,2./3.,3./4.)

# Builds the supercell filled up with any element
supercelula=crystal('H',[base],spacegroup=grupoespacial,cellpar=[pa,pa,pc,aa,ab,ac],size=(rx,ry,rz))

# Checks if the supercell has at least two atoms of each element
# Checks if the chemical formula can be properly reproduced with the amount of atoms in the supercell
totalelemento=[round(i*len(supercelula)) for i in compo.values()]
for i in totalelemento:
	if i<2:
		sys.exit('The chemical formula must produce at least two positions for each element in the supercell, change its size.')
if sum(totalelemento)!=len(supercelula):
	sys.exit('The chemical formula can\'t be properly reproduced in the supercell, change its size.')

# Fills up the supercell according to the given elements and their atomic fractions
# Creates a dictionary of elements and their indices in the supercell
limites=[0]
for i in range(len(totalelemento)):
	limites.append(limites[i]+totalelemento[i])
Lista=dict((i,[]) for i in compo.keys()) # dictionary of elements and their indices
for i in range(len(totalelemento)):
	for j in range(limites[i],limites[i+1]):
		supercelula[j].symbol=list(compo.keys())[i]
		Lista[supercelula[j].symbol].append(j)

# Elements in dictionary are sorted according to atomic amounts
Lista = dict(sorted(Lista.items(), key=lambda item: len(item[1])))

# Maximum possible free volume for each element (average volume)
# theoretical max free vol. is a fcc(hcp) packing of equal spheres
tavV=dict((i,supercelula.cell.volume/len(Lista[i])*0.74) for i in Lista)

# Function to find the distances of all atoms from a given atom regarding the same element 
def dists(indice):
	dists=supercelula.get_distances(indice,Lista[supercelula[indice].symbol],mic=True) # mic=True for periodic bondaries
	return dists

# Function to find the total length, all distances among atoms of the same element 
def totalL():
	totalL=0
	for i in range(len(supercelula)):
		totalL+=sum(dists(i))
	totalL=totalL/2 # since each distance is summed twice
	return totalL

# Function to find the total free length (minimum distance among atoms) regarding each element 
def freeL():
	freeL=0
	for i in range(len(supercelula)):
		freeL+=sorted(dists(i))[1] # the first distance is from the atom to itself (0), the second is the nearest neighbour
	return freeL

# Function to estimate the entropic force of an element, hereafter Entropic Force Parameter (EFP)
def el_entropia_for(elemento):
	el_entropia_for=0
	for i in Lista[elemento]:
		orddists=sorted(dists(i))
		# The calculated free volume here is a cube since we are interested in just estimating the entropic force 
		# by the total difference in volume regarding the maximum theoretical average volume (see tavV function above)
		v=orddists[1]**3 # the first distance is from the atom to itself (0), the second is the nearest neighbour
		# Summation of all volumetric differences regarding the theoretical max volume of the element
		el_entropia_for+=abs(v-tavV[supercelula[i].symbol])
	return el_entropia_for

# Function to find the total free volume (non overlapping spheres regarding each element)
def freeV():
	freeV=0
	for i in range(len(supercelula)):
		freeV+=4/3*np.pi*(sorted(dists(i))[1]/2)**3
	return freeV

# Function to swap elements (atomic positions) in the supercell according to their indices
def troca(indice1,indice2):
	temp=supercelula[indice1].symbol
	supercelula[indice1].symbol=supercelula[indice2].symbol
	supercelula[indice2].symbol=temp
	# Dictionary of elements and their positions must be updated
	Lista[supercelula[indice1].symbol][Lista[supercelula[indice1].symbol].index(indice2)]=indice1
	Lista[supercelula[indice2].symbol][Lista[supercelula[indice2].symbol].index(indice1)]=indice2

# Initial conditions
itl=totalL()
ifl=freeL()
ifv=freeV()
print('-- mcmaxent --')
print('Number of atoms: '+str(len(supercelula)))
print('Supercell Chemical Formula: ',end='')
for i in reversed(Lista):
	por = re.sub(r'\.0','',str(round(compo[i]*100,1))) # takes amount as percentage and removes .0
	print(i+por,end='')
print('\n')
print('Initial Total Length: '+str(itl))
print('Initial Free Length: '+str(ifl))
print('Initial Free Volume: '+str(ifv))
print('\b')

# Outputs the initial cif file
io.write(sys.argv[1]+'_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'_initial.cif',supercelula,'cif')
if ver:
	view(supercelula) # ase GUI output

# Main loop
# Shuffle positions and keep them allays the EFP gets smaller
numero_de_trocas=int(len(supercelula)**(3/2)) # number of shuffle steps, it is a guess
els=list(Lista.keys())
forca=dict((i,el_entropia_for(i)) for i in els) # EFP of each element
soma_forca=sum(forca.values()) # total EFP of the elements
soma_forca_depois=soma_forca # total EFP after swappings
i = 0
while i<=numero_de_trocas:
	i+=1
	# simple pair swap
	a=rnd.randint(0,len(supercelula)-1)
	b=a
	while b==a: # check if the second sorted position is the same or not
		b=rnd.randint(0,len(supercelula)-1)
	troca(a,b)
	forca=dict((i,el_entropia_for(i)) for i in els) # EFP of each element
	soma_forca_depois=sum(forca.values()) # total EFP of the elements
	if soma_forca_depois>=soma_forca:
		troca(a,b)
	else:
		soma_forca=soma_forca_depois
	# triple swap
	a=rnd.randint(0,len(supercelula)-1)
	b=a
	while b==a: # check if the second sorted position is the same or not
		b=rnd.randint(0,len(supercelula)-1)
	c=b
	while c==b or c==a: # check if the third sorted position is the same as a or b
		c=rnd.randint(0,len(supercelula)-1)
	troca(a,b)
	troca(b,c)
	forca=dict((i,el_entropia_for(i)) for i in els) # EFP of each element
	soma_forca_depois=sum(forca.values()) # total EFP of the elements
	if soma_forca_depois>=soma_forca:
		troca(b,c)
		troca(a,b)
	else:
		soma_forca=soma_forca_depois
		if verbose: 
			print('Triple')
	if verbose:
		print('\r Shuffle: ',str(i),' EFP: ',str(soma_forca),'           ',end='')
		

# Main loop end

#Voronoi analysis
# Function to produce a matrix with the supercell and its images for a given element (pbc mirrors)
def pbc(elemento):
	mirr=[supercelula[i].position for i in range(len(supercelula)) if supercelula[i].symbol==elemento]
	entrada=np.array(mirr)
	for x in [-supercelula.cell[0],np.array([0,0,0]),supercelula.cell[0]]:
		for y in [-supercelula.cell[1],np.array([0,0,0]),supercelula.cell[1]]:
			for z in [-supercelula.cell[2],np.array([0,0,0]),supercelula.cell[2]]:
				if not (sum(x)==0 and sum(y)==0 and sum(z)==0):
					soma=entrada+x+y+z
					mirr.extend([i for i in soma])
	return mirr
# dictionary of elements and Voronoi volumes of their atoms
vorvol=dict((i,[]) for i in Lista.keys())
for i in vorvol:
	vor=Voronoi(pbc(i))
	vorvol[i]=[ConvexHull(vor.vertices[vor.regions[vor.point_region[i]]]).volume for i in range(len(Lista[i]))]


# Final conditions
shtl=totalL()
shfl=freeL()
shfv=freeV()
print('\n')
print('Final Total Length: '+str(shtl))
print('Final Free Length: '+str(shfl))
print('Final Free Volume: '+str(shfv))

# Outputs the shuffle cif file
io.write(sys.argv[1]+'_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'_shuffle.cif',supercelula,'cif')
if ver:
	view(supercelula) # ase GUI output

# Statistics for each element
statistics=[] # lines of the statistics report
elFV=[] # free volumes of each atom regarding each element
statistics.append('Element\t| Theor. Free Vol.\t| Mean Free Vol.\t| Std. Deviation\t| Voronoi Mean Vol.\t| Std. Deviation')
for i in Lista:
	for j in Lista[i]:
		elFV.append(4/3*np.pi*(sorted(dists(j))[1]/2)**3)

	statistics.append(i+"\t| {0:8.12f}\t| {1:8.12f}\t| {2:8.12f}\t| {3:8.12f}\t| {4:8.12f}".format(tavV[i],np.mean(elFV),np.std(elFV),np.mean(vorvol[i]),np.std(vorvol[i])))
	elFV.clear()

# Statistics output
print('\b')
for i in statistics:
	print(i)
print('\b')

# Simmetry verification and space group determination
cell=(supercelula.get_cell(),supercelula.get_scaled_positions(),supercelula.get_atomic_numbers())
grupo=spglib.get_spacegroup(cell, symprec=1e-5)
print('Space Group: ',grupo)

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

if ver:
	view(supercelula) # ase GUI output

# Outputs the txt file
out = open(sys.argv[1]+'_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+'.txt', "a")
out.write('-- mcmaxent --'+'\n')
out.write(sys.argv[1]+' '+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+' '+str(datetime.now())+'\n')
out.write('Number of atoms:'+str(len(supercelula))+'\n')
out.write('Supercell Chemical Formula: ')
for i in reversed(Lista):
	subs = str(round(compo[i]*100,1))
	subs = re.sub(r'\.0','',subs) # removes .0
	out.write(i+str(subs))
out.write('\n')
out.write('Initial conditions'+'\n')
out.write('  Total length: '+str(itl)+'\n')
out.write('  Free length: '+str(ifl)+'\n')
out.write('  Free volume: '+str(ifv)+'\n')
out.write('Final conditions:'+'\n')
out.write('  Total length: '+str(shtl)+'\n')
out.write('  Free length: '+str(shfl)+'\n')
out.write('  Free volume: '+str(shfv)+'\n')
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

print('Exec. time: {}'.format(datetime.now()-t_inicio))
print('\b')
