#!/usr/bin/env python

# Quick Brute Force Maximum Entropy (qbfmaxent)
# april/2022 - march/2023

# ASE (Atomic Simulation Environment) python script to
# produce supercells of bcc,fcc or hcp lattices according to the
# maximum entropy principle (maximum free volume of each elment).
# Refs:
# S.Q. Wang, Entropy, 15 (2013), p. 5536
# A.D. Wissner-Gross, C.E. Freer, Phys. Rev. Lett., 110 (2013), p. 168702
# S.-M. Zheng, W.-Q. Feng, S.-Q. Wang, Comp. Mat. Sci., 142 (2018), p. 332
#
# The algorithm produces an initial supercell according to the input data (chemical formula, lattice and supercell size). 
# This initial supercell is randomily shuffled and after that the atomic positions are swapped until the global free volume
# of the elements is maximazed by keeping, at the same time, an even distribution of free volumes. The algorithm employs a
# brute force swapping, i.e., for a given element, all its positions are swapped with all other remaining elements. Many scans,
# i.e., swappings all over a given element positions are made until the global free volume remains unchanged near the evenly
# distribution of maximum free volume. In order to make the process much less time consuming the "Jar of rocks, pebbles and sand"
# principle is followed, i.e., larger free volumes go first. Firstly, the minor element in content has its free volume maximized
# by swaping its positions with all others, once it is done its positions keep unchanged. The same principle is applied to the
# second minor element in content, now swapping positions with all other remaining elements. This procedure goes on until the
# second major element in content. The major element in content just fills the remaining positions.

# HOW TO CITE  
# If you use this script in your work, please cite:  
# J.H. Mazo, C. Soares, G.K. Inui, M.F. de Oliveira, J.L.F. Da Silva, Materials Science and Engineering: A, 929 (2025), p. 148053  
# https://doi.org/10.1016/j.msea.2025.148053  

# Main python packages needed:
# ase: Atomic Simulation Environment (ASE) https://gitlab.com/ase/ase
# spglib: Software library for crystal symmetry search (Spglib) https://atztogo.github.io/spglib/
# 
# Both packages are also avaliable in Anaconda environment: 
# https://anaconda.org/conda-forge/ase
# https://anaconda.org/conda-forge/spglib  

# Command Line
#--------------------------
# $ python qbfmaxent.py --f <argv1> --b <argv2> --x <argv3> --y <argv4> --z <argv5> --b2_Cs <argv6> --b2_Cl <argv7> --n1 <argv8> --n2 <argv9> --verbose<optional> --view<optional>
#
# <argv1> = chemical formula in Hill notation
# <argv2> = base lattice (bcc, fcc, hcp or b2)
# <argv3>, <argv4>, <argv5> = natural numbers, repetitions of the base cell in x, y and z
# <argv6> = elements exclusively in Cs positions (without spaces)
# <argv7> = elements exclusively in Cl positions (without spaces)
# <argv8> = number of the first POSCAR file generated (optional)
# <argv9> = number of the last POSCAR file generated (optional)
# argv8 and argv9 are only used if you want to generated a great number of supercells at once, e.g. --n1 1 and --n2 100 will generate 100 supercells labeled from 1 to 100
# --verbose is optional if you want verbose output
# --view is optional if you want ASE-GUI for crystal visualization
# Use 'python qbfmaxent.py -h' for help.

# Outputs:
# - verbose running status on the fly
# - final structure(spacegroup), supercell parameters, atomic positions and statistics
# - files:
#   argv1_argv2argv3argv4argv5.txt
#		text file with the final results, structure(spacegroup), supercell parameters, statistics and atomic positions (appended file)
# 	argv1_argv2argv3argv4argv5_initial.cif
#		cif file with the initial supercell structure
#	argv1_argv2argv3argv4argv5_shuffle.cif
#		cif file with the supercell structure after the ramdomic shuffle
#	argv1_argv2argv3argv4argv5_spacegroup_final.cif
#		cif file with the supercell final structure
#
#	How to cite
#------------------------------------------------------------
#   Authors, Paper Title, Journal Name, Year. (TO BE DEFINED)
#------------------------------------------------------------

import sys
import numpy as np
import random as rnd
import spglib
import re
import argparse
from datetime import datetime
from ase.spacegroup import crystal
from ase.visualize import view
from ase.formula import Formula
from ase import io
from ase.build.tools import sort
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Quick Brute Force Maximum Entropy (qbfmaxent)')
    parser.add_argument('--f', type=str, nargs='?', default='Zr25Al25Ni25Cu25', metavar='Zr25Al25Ni25Cu25', help='Chemical formula in Hill notation')
    parser.add_argument('--b', type=str, nargs='?', default='fcc', choices=['fcc','bcc','hcp','b2'], metavar='fcc', help='Base Lattice: fcc, bcc, hcp or b2')
    parser.add_argument('--x', type=int, nargs='?', default='2', metavar='2', help='Number of unit cells in x')
    parser.add_argument('--y', type=int, nargs='?', default='2', metavar='2', help='Number of unit cells in y')
    parser.add_argument('--z', type=int, nargs='?', default='2', metavar='2', help='Number of unit cells in z')
    parser.add_argument('--verbose', nargs='?', default='n', help='If you want verbose output')
    parser.add_argument('--view', nargs='?', default='n', help='If you want ASE-GUI for crystal visualization')
    parser.add_argument('--b2_Cs', type=str, nargs='?', metavar='ZrAl', help='Elements EXCLUSIVELY in Cs positions of b2 (without spaces)')
    parser.add_argument('--b2_Cl', type=str, nargs='?', metavar='NiCu', help='Elements EXCLUSIVELY in Cl positions of b2 (without spaces)')
    parser.add_argument('--loop', nargs='?', default='n', help='if you want to generated diferent supercells in a single run')
    parser.add_argument('--n1', type=int, nargs='?', default='1', metavar=1, help='Label of the first supercell generated')
    parser.add_argument('--n2', type=int, nargs='?', default='1', metavar=1, help='Label of the last supercell generated')
    return parser.parse_args()


# atomic radii (angstroms)
# from: Miracle, D. B.; Sanders, W. S.; Philosophical Magazine, v. 83, n. 20, p. 2418, 2003.
raios={'Ag':1.42,'Al':1.43,'Au':1.45,'B':0.78,'Be':1.12,'Bi':1.60,'C':0.77,'Ca':1.97,'Ce':1.82,'Co':1.28,'Cr':1.28,'Cu':1.27,'Dy':1.77,'Er':1.76,
	   'Fe':1.28,'Ga':1.32,'Gd':1.74,'Ge':1.14,'Hf':1.67,'La':1.87,'Mg':1.6,'Mn':1.32,'Mo':1.39, 'Nb':1.46,'Nd':1.85,'Ni':1.28,'P':1.0,
	   'Pb':1.75,'Pd':1.41,'Pt':1.38,'Rh':1.34,'Si':1.02,'Sn':1.62,'Ta':1.49,'Th':1.8,'Ti':1.46,'U':1.58,'V':1.34,'Y':1.8,
	   'Zn':1.38,'Zr':1.58}

print('\b')

# Initial time
t_inicio=datetime.now()

# Input data
args = parse_arguments()
compo=Formula(args.f).count() # element symbols and atomic amounts from the chemical formula in Hill notation
gruponumero={'bcc':229,'fcc':225,'hcp':194,'b2':221} # space group numbers
grupoespacial=gruponumero[args.b] 
rx=args.x # base cell repetitions in x
ry=args.y # base cell repetitions in y
rz=args.z # base cell repetitions in z
if rx<1 or ry<1 or rz<1:
	sys.exit('Supercell must be at least 1x1x1')
verbose=False
if args.verbose!='n':
	verbose=True # activates the verbose output
ver=False
if args.view!='n':
	ver=True # activates the ase GUI crystal viewer

# Converts the atomic amounts into fractions
compo=dict(zip(compo.keys(),[i/sum(compo.values()) for i in compo.values()]))

# Estimative of the base cell parameter, in angstroms,
# calculated from a weighted average of the atomic radii 
rmedio=sum(list(compo.values())*np.array([raios[i] for i in compo.keys()]))
if grupoespacial==229:
	pa=4*rmedio/(3**(1/2)) # base cell parameter for bcc
	pc=pa; aa=90; ab=90; ac=90; base=[(0,0,0)]
	elsin=('H')
elif grupoespacial==225:
	pa=4*rmedio/(2**(1/2)) # base cell parameter for fcc
	pc=pa; aa=90; ab=90; ac=90; base=[(0,0,0)]
	elsin=('H')
elif grupoespacial==194:
	pa=2*rmedio # base cell parameter for hcp       
	pc=pa*(8/3)**(1/2) # ideal c/a ratio
	aa=90; ab=90; ac=120; base=[(1./3.,2./3.,3./4.)]
	elsin=('H')
elif grupoespacial==221:
	pa=4*rmedio/(3**(1/2)) # base cell parameter for b2
	pc=pa; aa=90; ab=90; ac=90; base=[(0,0,0),(.5,.5,.5)]
	elsin=('Cs','Cl')

# Builds the supercell filled up with a prototype
# Important: for the b2 strucutre the even indices are Cs positions and the odd indices are Cl positions 
supercelula=crystal(elsin,base,spacegroup=grupoespacial,cellpar=[pa,pa,pc,aa,ab,ac],size=(rx,ry,rz))

# Checks if the supercell has at least two atoms of each element
# Checks if the chemical formula can be properly reproduced with the amount of atoms in the supercell
totalelementos=dict(zip(compo.keys(),[round(i*len(supercelula)) for i in compo.values()]))
for i in totalelementos.values():
	if i<2:
		sys.exit('The chemical formula must produce at least two positions for each element in the supercell, change its size.')
if sum(totalelementos.values())!=len(supercelula):
#The best way to minimize relative changes in proportions is to modify the most occurring element adding or removing 1 atom
	if sum(totalelementos.values())<len(supercelula):
		totalelementos[max(totalelementos,key=totalelementos.get)]+=1
	if sum(totalelementos.values())>len(supercelula):
		totalelementos[max(totalelementos,key=totalelementos.get)]-=1
	if sum(totalelementos.values())!=len(supercelula):
		sys.exit('The chemical formula can\'t be properly reproduced in the supercell, change its size.')
#If the user inputs the b2 structure set occupations
if grupoespacial==221:
        evenelements = Formula(args.b2_Cs).count()
        for i in evenelements.keys():
                evenelements[i]=totalelementos[i]
        #check if evenelements fit even positions
        if sum(evenelements.values())>len(supercelula)/2:
                evenelements[max(evenelements,key=evenelements.get)]-=1
        if sum(evenelements.values())<len(supercelula)/2:
                evenelements[max(evenelements,key=evenelements.get)]+=1
        oddelements = Formula(args.b2_Cl).count()
        for i in oddelements.keys():
                oddelements[i]=totalelementos[i]
        #check if oddelements fit odd positions
        if sum(oddelements.values())>len(supercelula)/2:
                oddelements[max(oddelements,key=oddelements.get)]-=1
        if sum(oddelements.values())<len(supercelula)/2:
                oddelements[max(oddelements,key=oddelements.get)]+=1

# Fills up the supercell according to the given elements, their atomic fractions and preferential sites
# Creates a dictionary of elements and their indices in the supercell
Lista=dict((i,[]) for i in compo.keys()) # dictionary of elements and their indices
if grupoespacial!= 221:
	k=0
	for i in totalelementos.keys():
		for j in range(totalelementos[i]):
			supercelula[k].symbol=i
			Lista[supercelula[k].symbol].append(k)
			k+=1
else:
	#fill even positions
	keven=0
	for i in evenelements.keys():
		for j in range(evenelements[i]):
			supercelula[keven].symbol=i
			Lista[supercelula[keven].symbol].append(keven)
			keven+=2
	#fill odd positions
	kodd=1
	for i in oddelements.keys():
		for j in range(oddelements[i]):
			supercelula[kodd].symbol=i
			Lista[supercelula[kodd].symbol].append(kodd)
			kodd+=2
	#fill remaining positions
	copia = totalelementos.copy()
	for i in evenelements.keys():
		del copia[i]
	for i in oddelements.keys():
		del copia[i]
	for i in copia.keys():
		for j in range(copia[i]):
			if keven<=len(supercelula)-2:
				supercelula[keven].symbol=i
				Lista[supercelula[keven].symbol].append(keven)
				keven+=2
			else:
				supercelula[kodd].symbol=i
				Lista[supercelula[kodd].symbol].append(kodd)
				kodd+=2

# Elements in dictionary are sorted according to atomic amounts
# in order to follow the "Jar of rocks, pebbles and sand" principle 
Lista = dict(sorted(Lista.items(), key=lambda item: len(item[1])))

# Maximum possible free volume for each element (average volume)
# theoretical max free vol. is a fcc(hcp) packing of equal spheres
tavV=dict((i,supercelula.cell.volume/len(Lista[i])*0.74) for i in Lista)

# Initial conditions
itl=totalL()
ifl=freeL()
ifv=freeV()
print('-- qbfmaxent --')
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
io.write(args.f+'_'+args.b+'_'+str(args.x)+str(args.y)+str(args.z)+'_initial.cif',supercelula,'cif')
supercelula_sorted=sort(supercelula)
io.write('POSCAR_initial',supercelula_sorted,'vasp')

if ver:
	view(supercelula) # ase GUI output

for z in range(args.n1,args.n2+1):
	# Shuffle to start from a more randomic distribution
	numero_de_trocas=len(supercelula)**2 # number of shuffle steps, the square of atoms in the supercell
	# shfv=freeV()
	for i in range(numero_de_trocas):
		a=rnd.randint(0,len(supercelula)-1)
		b=a
		while b==a: # check if the second sorted position is the same or not
			b=rnd.randint(0,len(supercelula)-1)
		if grupoespacial==221:
			if (a%2==0 and b%2==0) or (a%2!=0 and b%2!=0):  
				troca(a,b)
				if verbose:
					print('\r Shuffle: ',str(i),' (',str(a),'-',str(b),')      ',end='')
		else:
			troca(a,b)
			if verbose:
				print('\r Shuffle: ',str(i),' (',str(a),'-',str(b),')      ',end='')

	# Conditions after shuffle
	shtl=totalL()
	shfl=freeL()
	shfv=freeV()
	print('\n')
	print('Total Length (after shuffle): '+str(shtl))
	print('Free Length (after shuffle): '+str(shfl))
	print('Free Volume (after shuffle): '+str(shfv))

	# Outputs the shuffle cif file
	supercelula_sorted=sort(supercelula)
	io.write('POSCAR_shuffle',supercelula_sorted,'vasp')
	io.write(args.f+'_'+args.b+'_'+str(args.x)+str(args.y)+str(args.z)+'_shuffle.cif',supercelula,'cif')
	if ver:
		view(supercelula) # ase GUI output

	# Preparing the main loop to minimize the EFP
	# Scans the element lists and positions following the "Jar of rocks, pebbles and sand" principle
	# i.e., larger free volumes goes first
	els=list(Lista.keys())
	forca=dict((i,el_entropia_for(i)) for i in els) # EFP of each element
	soma_forca=sum(forca.values()) # total EFP of the elements
	soma_forca_depois=soma_forca # total EFP after swappings
	digs=re.sub(r'.','0',str(len(supercelula)-1)) # to use with verbose

	# Main loop to minimize the EFP, i. e., maximize the free volume
	# It scans the element lists and their atomic positions following the "Jar of Rocks and pebbles" principle
	# (larger free volumes goes first). It minimizes the entropic force firstly for the minor element in content
	# by swaping positions with all others. Once the first element is done its positions keep unchanged. 
	# The same principle is applied to the second minor element in content and so on.
	for i in range(len(els)-1):
		vai=True # flag to start a new scan
		v=0 # scanning number
		forca_antes=forca[els[i]] # total EFP of the element
		forca_depois=forca_antes # total EFP after swapping
		last='              ' # used with verbose option
		if verbose:
			print('\nTotal EFP: ',str(sum(forca.values())))
		while vai:
			v=v+1 # increments the scanning number
			a_mat=Lista[els[i]] # atomic positions of the first element, a
			for j in range(i+1,len(els)):
				b_mat=Lista[els[j]] # atomic positions of the second element, b
				for k in range(len(a_mat)): # scans all a positions
					a=a_mat[k]
					for h in range(len(b_mat)): # scans all b positions
						b=b_mat[h]
						forca_antes=forca[els[i]] # EFP of element a before swapping
						if grupoespacial==221:
							if (a%2==0 and b%2==0) or (a%2!=0 and b%2!=0): 
								troca(a,b) # swapps the positions of given a and b
						else:
							troca(a,b) # swapps the positions of given a and b
						forca_depois=el_entropia_for(els[i]) # new EFP of element a
						if forca_depois>=forca_antes: # if the EFP is not reduced then reverse the swapping
							if grupoespacial==221:
								if (a%2==0 and b%2==0) or (a%2!=0 and b%2!=0):
									troca(a,b)
							else:
								troca(a,b) 
						else: # otherwise updates the EFP of elements in positions a, b and updates the lists
							forca[els[i]]=forca_depois
							forca[els[j]]=el_entropia_for(els[j])
							a_mat=Lista[els[i]] # updates a list
							b_mat=Lista[els[j]] # updates b list
							last='('+str(v)+' '+els[i]+digs[:-len(str(k))]+str(k)+'-'+els[j]+digs[:-len(str(h))]+str(h)+')' # used with verbose option
						if verbose:
							print('\rElement:',els[i],' Scan:',str(v),' Swap:',els[i]+digs[:-len(str(k))]+str(k)+'-'+els[j]+digs[:-len(str(h))]+str(h),' EFP:',str(forca[els[i]]),last,'           ',end='')
			
			soma_forca_depois=sum(forca.values()) # total entropic force of all elements after a and b swappings
			if soma_forca==soma_forca_depois:
				vai=False # if the total EFP remains the same as the previous scan then interrupts (breaks the element loop)
			else:
				soma_forca=soma_forca_depois # otherwise updates the total EFP and keeps the scans (keeps the element loop)
		if ver:
				view(supercelula)
	if verbose:
				print('\nTotal EFP: ',str(soma_forca_depois))

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

	# Final conditions after EFP minimization	
	print('\b')
	print('Final Total Length: '+str(totalL()))
	print('Final Free Length: '+str(freeL()))
	print('Final Free Volume: '+str(freeV()))

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
	supercelula_sorted=sort(supercelula)
	io.write(args.f+'_'+args.b+'_'+str(args.x)+str(args.y)+str(args.z)+'_'+hm+'_final.cif',supercelula,'cif')
	# Outputs a POSCAR file for VASP
	io.write('POSCAR'+str(z),supercelula_sorted,'vasp')

	if ver:
		view(supercelula) # ase GUI output

	# Outputs the txt file
	out = open(args.f+'_'+args.b+'_'+str(args.x)+str(args.y)+str(args.z)+'.txt', "a")
	out.write('-- qbfmaxent --'+'\n')
	out.write(args.f+' '+args.b+' '+str(args.x)+str(args.y)+str(args.z)+' '+str(datetime.now())+'\n')
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
	out.write('After shuffle:'+'\n')
	out.write('  Total length: '+str(shtl)+'\n')
	out.write('  Free length: '+str(shfl)+'\n')
	out.write('  Free volume: '+str(shfv)+'\n')
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

	print('Exec. time: {}'.format(datetime.now()-t_inicio))
	print('\b')
	f=open("Final_free_volumes","a")
	f.write(str(z) + "	" + str(freeV()) + "\n")
	f.close()
