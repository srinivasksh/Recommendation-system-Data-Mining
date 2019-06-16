import timeit
from numpy import *
import pandas as pd
import itertools
from collections import OrderedDict
from subprocess import check_output

start_time = timeit.default_timer()

## Parameters used in the code
delta_const = 0.5
global_min_sup = 0.05
phi_const = 0.05

## Dictionary containing support for all items and itemsets
ck_dict = {}

## Output List containing the frequent items 
## Note: This doesnt include items with k=1 as they are obvious
frequent_itemset = []

##Read input dataset and create a dictionary of items and their support
def msapriori_algo(dataset):
	c1= {}
	for tran in dataset:
		tran=tran.split()
		for item in tran:
			item=int(item)
			if item in c1:
				c1[item] += 1
			else:
				c1[item] = 1
	return c1

## Check for support difference constraint
## For the items in input itemset, calculate Max(MIS) - Min(MIS) 
def check_support_diff_const(list_ip):
	tmp_list = dict((i,item_mis[i]) for i in list_ip)
	support_diff_val = (max(tmp_list.values()) - min(tmp_list.values()))
	return(support_diff_val)		
	
## Candidate generation for Level2 (K=2)
## Logic is to create pairs of items from k=1 (L)
## and pick those that meet support different constraint 	
def level2_candidate_gen(list_ip,phi_val):
	item_sets = list(itertools.combinations(list_ip,2))
	ck_tmp = []
	for i in item_sets:
		diff = check_support_diff_const(i)
		if (diff <= phi_val):
			ck_tmp.append(i)
	return(ck_tmp)

## Check for pruning (subsets to be part of previous iteration's freq set)
def ms_candidate_pruning(ck_ip,num):
	valid_ck = 'TRUE'
	cand_tmp = set(list(itertools.chain(ck_ip)))
	cand_subset = list(itertools.combinations(cand_tmp,num-1))
	for c in cand_subset:
		if c not in ck_dict:
			valid_ck = 'FALSE'
	return valid_ck

## Candidate generation for Level3 and above (K>2)
## Logic is to create pairs of items from previous freq sets(Fk-1)
## and pick those that meet support different constraint and
## those that pass pruning (subset to be part of Fk-1)	
def ms_candidate_gen(fk_ip,phi_val,num):
	ck_tmp = set()
	for i in fk_ip:
		for j in fk_ip:
			if len(set(i).union(set(j))) == num:
				ck_union = set(i).union(set(j))
				if (check_support_diff_const(ck_union) <= phi_val):
					if (ms_candidate_pruning(ck_union,num)):
						ck_tmp.add(frozenset(ck_union))
	return ck_tmp

## Function to sort a dictionary based on value (MIS)
def sort_dict_by_values(dict_in,order):
	if order=='A':
		dict_tmp = OrderedDict(sorted(dict_in.items(), key=lambda x: x[1]))
	else:
		dict_tmp = OrderedDict(sorted(dict_in.items(), key=lambda x: x[1], reverse='TRUE'))
	dict_out = {}
	for k,v in dict_tmp.items():
		dict_out[k] = v
	return dict_out
	
## Generate Fk set based on Ck
def generate_fk_set(ck_dict_loop):
	ck_dict_sorted = sort_dict_by_values(ck_dict_loop,'A')
	fk_out = []
	for k,v in ck_dict_sorted.items():
		if (ck_dict_sorted[k]/tran_cnt >= item_mis[k[0]]):
			fk_out.append(k)
	return(fk_out)
	
## Main function
f = open('retail1.txt', "r")
dataset = f.readlines()
item_support = msapriori_algo(dataset)
tran_cnt = size(dataset)

##Local support relative to the number of transactions
LSRelative = global_min_sup

## Create a dictionary of item and its corresponding MIS
item_mis = {}
for key in item_support:
	item_mis[key] = (item_support[key]/tran_cnt)*delta_const
	if item_mis[key] < LSRelative:
		item_mis[key] = LSRelative

## Calculate min MIS to be used for calculation of Level1 Candidates
min_MIS = min(item_mis.values())

## Sort items by MIS value
item_sorted = sorted(item_mis, key=item_mis.get)

## Calculate L set (candidates for k=1)
L_List = []
for idx in item_sorted:
	if (item_support[idx]/tran_cnt > min_MIS):
		L_List.append(idx)

## Sort the candidate key list
L_List.sort()

## Calculate Frequent itemset for k=1
F1_List = []
for idx in L_List:
	if (item_support[idx]/tran_cnt > item_mis[idx]):
		idx_ip = []
		idx_ip.append(idx)
		F1_List.append(idx)

print("Found " + str(len(F1_List)) + " Frequent item sets for length: 1")
print(" ")
frequent_itemset.append(F1_List)

## Loop to calculate Frequent itemsets from k=2 until Fk is NULL
k = 2
end_of_loop = 0
while (end_of_loop == 0):
	print("Calculating Frequent Item sets calculation of size: " + str(k) + " ...")
	if (k==2):
		ck = level2_candidate_gen(L_List,phi_const)
	else:
		ck = ms_candidate_gen(fk,phi_const,k)

	fk = []
	ck_dict_curr = {}
	for c in ck:
		ck_dict[c] = 0
		ck_dict_curr[c] = 0
		c_set = set(c)
		for tran in dataset:
			tran_set = tran.split()
			tran_set = list(map(int,tran_set))
			if (c_set <= set(tran_set)):
				ck_dict[c] += 1
				ck_dict_curr[c] += 1
	if len(ck_dict_curr) > 0:
		fk = set(generate_fk_set(ck_dict_curr))
		fk = list(map(list,fk))
	if len(fk) > 0: 
		frequent_itemset.append(fk)
		print("Found " + str(len(fk)) + " Frequent item sets for length: " + str(k))
		print("                                                             ")
		k = k+1
	else:
		print("No Frequent Itemsets found for size: " + str(k))
		print("                                          ")
		print("Below are the Frequent Itemsets for the input dataset: ")
		print("                                          ")
		for elem in frequent_itemset:
			if len(elem) > 1:
				for elem_sub in elem:
					print([elem_sub])
			else:
				print(elem)
		print(" ")
		elapsed_time = timeit.default_timer()-start_time
		print("Total time taken for MSApriori algorithm (without item constraint): " + str(elapsed_time))
		print(" ")
		print("End of code!")
		end_of_loop = 1