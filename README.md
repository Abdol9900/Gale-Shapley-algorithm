 # Gale-and-Shapley-algorithm

1 Generating Male and female number of lists.
To run the code Generating-men_list.py, one needs python 3.6 installed, along with the packages, the code generated a number of lists for the male. As well as to run the code Generating-women list is exactly the same as running the code to generate the men list. In addition, second needs are creating two text files one named men_list.txt and another one named women_list.txt First ensure the following packages are installed and imported.
Listing 1: Generating-men_list.py - Import Statements import random import json
Listing 2: Generating-men_list.py- Generating number for male and saved in text file data = {} dictionaryKeyLength = 50 arrayListLength = 50
def shuffleArray(length): baseList = [] for index in range(length): baseList.append(str(index + 1))
random.shuffle(baseList) return baseList
for index in range(dictionaryKeyLength): data[str(index + 1)] = shuffleArray(arrayListLength)
with open('./men_list.txt' , 'w') as f: f.write(json.dumps(data)) # saved numbers in men_list.txt
Listing 2: Generating-men_list.py- Generating number for female and saved in text file data = {} dictionaryKeyLength = 50 arrayListLength = 50
def shuffleArray(length): baseList = [] for index in range(length): baseList.append(str(index + 1))
random.shuffle(baseList)
return baseList
for index in range(dictionaryKeyLength): data[str(index + 1)] = shuffleArray(arrayListLength)
with open('./women_list.txt' , 'w') as f: f.write(json.dumps(data)) # saved numbers in women_list.txt
2 Gale shapely algorithm
3 1. In the loop in the first round
4 a. every single man proposes to the most preferred woman.
5 b. the answer of every woman to the man she prefers most is "maybe" while "no" to the rest of people. As such, the woman becomes temporarily "engaged" to the most preferred man, and equally, that man is temporarily engaged to her.
6 2. In every succeeding round
7 a. every single man proposes to the most-liked woman to whom he has not yet proposed even if the woman is engaged.
8 b. if the woman is not presently engaged or likes the guy over her current temporary man, she replies "maybe". Hence, she rejects her present temporary partner who becomes single.
9 This process continues until everybody is engaged.
10 To run the code G-shapely.py one needs to ensure that python packages are installed and imported in order to run this code.
Listing 1: G-shapely.py- imported statement
import collections import json import matplotlib.pyplot as plt import networkx as nx import pprint
from copy import deepcopy
pp = pprint.PrettyPrinter() # Pretty printer
Listing2: G-shapely.py – reading the data from the two text files
if __name__=='__main__': global _men, _women, men_cp, women_cp
f_men = open('men_list.txt') _men = json.load(f_men) f_women = open('women_list.txt') _women = json.load(f_women)
Listing3: G-shapely.py - creating a copy of the male and female lists
#Create copies of preference list for further operations men_cp = deepcopy(_men) women_cp = deepcopy(_women)
# Append characters to male preference list _men = m(_men)
# Append characters to male preference list _women = f(_women)
Listing 4: G-shapely.py- function that allows to select to start with male propose to female or female propose to male.
print('1. Should Men propose first?') print('2. Should Women propose first?') print('Enter your choice(1/2)')
Listing 5: G-shapely.py - 1st choice and starting with male propose to female.
x = int(input())
if x == 1: __initMen__() stable_matching_men() matched_pairs = strip_m(tmp) matched_pairs.sort(key = lambda x: x[0]) print('Following are the stable pairs:') pp.pprint(matched_pairs)
male_optimal_cost = calc_cost(tmp) print("Total Cost calculated: {}".format(male_optimal_cost))
print("Women shortlist:") sw = shortlist_women(women_cp, tmp) pp.pprint(sw)
print("Men shortlist:") mw = shortlist_men(men_cp, sw, tmp) _mw = dict(mw)
cv = {} for k, v in sorted(_mw.items()): cv[k] = v
da = cv db = men_cp ordered_mw = reorder(da , db) pp.pprint(ordered_mw)
Listing 6: G-shapely.py- Define the male and female dictionary and my list
def m(mydict): """This function appends the character 'm' to the keys
of the male preference list and 'w' to all the items in the value.
Parameters ---------- mydict: A dict. The male preference lists
Returns ---------- A modified dict.
"""
return {k + 'm': [i + 'w' for i in v] for k, v in mydict.items()}
def f(mydict): """This function appends the character 'w' to the keys of the female preference list and 'm' to all the items in the value.
Parameters ---------- mydict: A dict. The female preference list
Returns ---------- A modified dict. """
return {k + 'w': [i + 'm' for i in v] for k, v in mydict.items()}
def strip_m(mylist): """A utility function to strip the unwanted characters from the resulting stable pairs.
Parameters ---------- mylist: A list. All the resulting stable pairs.
Returns ---------- The required stable pairs. """
z = [item for sublist in mylist for item in sublist]
p = [] for i in range(0, len(z) - 1, 2): p.append([z[i][:-1], z[i + 1][:-1]]) return p
tmp = [] free_men = []
free_women = []
Listing 7: G-shapely.py- define the preference lists for men and women
def __initMen__(): """We initialise the method, so we can pair every men from the given preference list. """
for i in _men.keys(): free_men.append(i)
def __initWomen__(): """We initialise the method, so we can pair every women from the given preference list. """ for i in _women.keys(): free_women.append(i)
Listing 8: G-shapely.py- method that allows men to propose to women
def men_propose(man): """This method will be run when we need men to propose first to the woman according to preference list.
Parameters ---------- man: An int. This represents the man in the freemen list. """
for woman in _men[man]:
# Boolean for whether a woman is taken or not taken_match = [couple for couple in tmp if woman in couple]
if (len(taken_match) == 0): # Engage the man and the woman tmp.append([man, woman]) free_men.remove(man) break
elif (len(taken_match) > 0): # Check ranking of the current male and the ranking of the 'to-be' male current_guy = _women[woman].index(taken_match[0][0]) potential_guy = _women[woman].index(man)
if (current_guy < potential_guy): z = taken_match[0][0] else: # The new guy is engaged free_men.remove(man)
# The previous male is now single free_men.append(taken_match[0][0])
# Update the fiance of the woman (tentatively) taken_match[0][0] = man break
Listing 9: G-shapely.py- method that allows women to propose to men.
def women_propose(woman): """This method will be run when we need women to propose first to the man according to preference list.
Parameters ---------- woman: An int. This represents the woman in the freewomen list. """ for man in _women[woman]:
# Boolean for whether a woman is taken or not taken_match = [couple for couple in tmp if man in couple]
if (len(taken_match) == 0): # Engage the man and the woman tmp.append([woman, man]) free_women.remove(woman) break
elif (len(taken_match) > 0): # Check ranking of the current female and the ranking of the 'to-be' female current_woman = _men[man].index(taken_match[0][0]) potential_woman = _men[man].index(woman)
if (current_woman < potential_woman): z = taken_match[0][0] else: # The new woman is engaged free_women.remove(woman)
# The previous female is now single free_women.append(taken_match[0][0])
# Update the better half of the man (tentatively) taken_match[0][0] = woman break
def stable_matching_men(): """ Function to run the algorithm when men propose first until the pairs are matched. """
while (len(free_men) > 0):
for i in free_men: men_propose(i)
def stable_matching_women(): """ Function to run the algorithm when women propose first until the pairs are matched. """
while (len(free_women) > 0): for i in free_women: women_propose(i)
Listing 10: G-shapely.py- method to enlist male list and female list into a single list.
def men_list(_men_): """ We need to use the list values from the male preference dict. This function enlists them into a single list.
Parameters ---------- _men_: A dict. Male preference dict
Returns ---------- A list of lists """
d = [] for k, v in _men_.items(): d.append(v)
return d
def women_list(_women_): """ We need to use the list values from the female preference dict. This function enlists them into a single list.
Parameters ---------- _women_: A dict. Female preference dict
Returns ---------- A list of lists """
d = [] for k, v in _women_.items(): d.append(v)
return d
3 The cost functions
Once every male and female present in the list are paired up, we create a list of men proposed to women and sum up the cost function. Finding out the cost works as follows: To find the cost function and we numbered the male preference list and the female preference list, In the male shortlist, we find out the position of female and likewise find out the position and then we calculated the total cost for male and female.
Listing 1: G-shapely.py- calculating the cost function.
ef calc_cost(tmp): """ This method calculates the total cost that is incorporated into building male and female optimal pairs.
Parameters ---------- tmp: A list. This f-list contains the male and female optimal solution in an unordered way.
Returns ---------- total cost: An int. This is the total cost calculated as per the cost function """
y = [] for i in range(len(_men)): y.append(men_list(_men)[i])
pairs = strip_m(tmp) pairs.sort(key = lambda x: x[0])
men_list_numbered = []
for s in y: s = [w.replace('w', '') for w in s] men_list_numbered.append(s)
cost_i = [] for i in range(len(pairs)): cost_i.append(men_list_numbered[i].index(pairs[i][1]) + 1)
z = []
for i in range(len(_women)): z.append(women_list(_women)[i])
pairs_for_women = strip_m(tmp) pairs_for_women.sort(key = lambda x: x[1])
women_list_numbered = []
for s in z: s = [w.replace('m', '') for w in s] women_list_numbered.append(s)
cost_j = [] for i in range(len(pairs_for_women)): cost_j.append(women_list_numbered[i].index(pairs_for_women[i][0]) + 1)
cost_list = cost_i + cost_j total_cost = sum(cost_list)
return total_cost
4 Shortlists
We followed two central implications of this initial proposal sequence to create a shortlist:
- If m recommends w, then there exists no stability in matching and it happens that m has a superior spouse than w.
- If w is proposed to by m, then there exists instability in matching where w has a poorer spouse than m.
From the above fundamentals, we can eliminate m from w’s list and w from m’s list if w gets an offer from someone she prefers to m. We apply this rule in a loop and the resulting list is called a shortlist.
Listing 1: G-shapely.py-creating a shortlist for male and female
def shortlist_women(women_cp, tmp): """ Creates shortlists for women preference list.
Parameters ---------- women_cp: A dict. Copy of original women preference list. tmp: A list. A list of optimal matchings.
Returns ---------- women_shortlist: A dict. Final shortlist for women. """
strip_m(tmp) tmp.sort(key = lambda x: x[0])
women_shortlist = {}
i = 0 for k, v in women_cp.items(): x = tmp[i][1][:-1] women_shortlist[x] = (women_cp.get(x))[:(women_cp.get(x)).index(tmp[i][0][:-1]) + 1]
i += 1
return women_shortlist
def shortlist_men(men_cp, sw, tmp): """ Creates shortlists for women preference list.
Parameters ---------- men_cp: A dict. Copy of original women preference list. sw: A dict. Women shortlist. tmp: A list. A list of optimal matchings.
Returns ---------- men_shortlist: A dict. Final shortlist for men. """
strip_m(tmp) tmp.sort(key = lambda x: x[0])
men_shortlist = collections.defaultdict(list) women_list = []
for k, v in sw.items(): for j in v: men_shortlist[j].append(k)
return men_shortlist
Listing 2: G-shapely.py- method to restore the order of preferences in any ranking list.
ef reorder(da, db): """This method restores the order of preferences in any ranking list
Parameters ---------- da: A dict. Unordered shortlist. db: A dict. Original preference list """
n = len(da) check = [0] * (n + 1) final = collections.defaultdict(list)
for k, v in da.items(): for j in range(n + 1): check[int(j)] = 0 for i in v: check[int(i)] = 1 for j in db[k]: if check[int(j)] == 1: final[k].append(j) return dict(final)
5 Rotation
The significance of rotation lies in the fact that if in the male optimal solution, each mi exchanges his partner wi for wi+1 (i + 1 mod n) then the resulting matching is also stable. The same can be stated for a female optimal solution. Once a rotation has been identified, we eliminated it. We do this in a loop so that one or more, so the rotations may be exposed in the resulting further reduced lists. Repeating this over a loop and pairing each man with the first woman in his reduced list yields a stable matching.
Listing 1: G-shapely.py- finding the collection of rotation. # auxiliary variables for finding rotations rotations = [] last_rotation_for_man = {} weight_of_rotation = {} level_of_rotation = {} level = 0 ordered_mw_copy = ordered_mw.copy()
G = nx.DiGraph() labels = {}
# Loop that goes throught the levels of rotations until it doesn't # find any more rotations while True: # Go through the unvisited nodes and find cycles exposed_rotations = [] visited = set() for k in ordered_mw.keys(): if not k in visited: current = k rotation = [] while current not in visited: visited.add(current) # A man with only one woman in his shortlist was found # Hence we won't be able to find any rotation in this path if len(ordered_mw[current]) < 2:
rotation.clear() break # add a tuple with (man, current partner, next partner) rotation.append((current, ordered_mw[current][0], ordered_mw[current][1])) # move to next man current = sw[ ordered_mw[current][1] ][-1]
while len(rotation) > 0 and rotation[0][0] != current: rotation.pop(0)
# add to list of rotations if a rotation was found if len(rotation) > 0: exposed_rotations.append(rotation)
# there aren't any more rotations if len(exposed_rotations) == 0: break
Listing 2: G-shapely.py- Loop that eliminates the rotations and calculates the weight of rotation # Loop that eliminates the rotations and calculates the weight of # each rotation for rotation in exposed_rotations: rotations.append(rotation) current_rotation = len(rotations) level_of_rotation[current_rotation] = level weight = 0
G.add_node(current_rotation) labels[current_rotation] = 'p{}'.format(current_rotation)
for man, woman, new_woman in rotation: weight += men_cp[man].index(woman) - men_cp[man].index(new_woman) weight += women_cp[woman].index(man) weight -= women_cp[new_woman].index(man)
if man in last_rotation_for_man: predecessor = last_rotation_for_man[man] G.add_edge(predecessor, current_rotation)
last_rotation_for_man[man] = current_rotation
if ordered_mw[man][0] == woman: ordered_mw[man].pop(0)
sw[new_woman].pop(-1) while sw[new_woman][-1] != man: x = sw[new_woman][-1] sw[new_woman].pop(-1) # TODO: check if new_woman in ordered_mw[x]: ordered_mw[x].remove(new_woman)
weight_of_rotation[current_rotation] = str(weight)
level += 1
6 Digraph Visualization
Listing 1: G-shapely.py- creating a directed graph by using NetworkX package.
# store list of edges in the graph before starting to delete some of them graph_edges = list(G.edges())
# Loop to delete unnecessary edges in the graph # we try to remove edges to see if removing them disconnects the nodes # that they're connecting for e in graph_edges: G.remove_edge(e[0], e[1])
if not nx.has_path(G, e[0], e[1]): G.add_edge(e[0], e[1])
# Debug code to print the rotations and the weights for i, rotation in enumerate(rotations): rotation_str = "" for man, woman, new_woman in rotation: rotation_str += "({}, {}),".format(man, woman) rotation_str = rotation_str[:-1]
print("p{} = {}".format(i + 1, rotation_str)) print("weight = {}".format(weight_of_rotation[i + 1]))
Listing 2: G-shapely.py- directed graph (networkX layout).
# Approximate distances that will be used to plot the graph # By default, if there is not a path between nodes the distance will be # the difference in their levels dist_dict = {} for i in range(1, 1 + len(rotations)): dist_dict[i] = {} for j in range(1, 1 + len(rotations)): dist_dict[i][j] = 1 + abs(level_of_rotation[i] - level_of_rotation[j])
# Compute shortest path and store in distance matrix for row, data in nx.shortest_path_length(G): for col, dist in data.items(): dist_dict[row][col] = dist dist_dict[col][row] = dist
# Set up the layout for plotting the graph pos = nx.kamada_kawai_layout(G, dist=dist_dict) nx.draw_networkx_nodes(G, pos, node_size=500) nx.draw_networkx_nodes(G, pos, nodelist=removed_rotations, node_color='b', node_size=500, alpha=0.4) nx.draw_networkx_edges(G, pos, node_size=500) nx.draw_networkx_labels(G, pos, labels, font_size=8)
# Increase the y coordinate a bit to plot the weights of nodes # above the nodes for k, v in pos.items(): coord = v coord[1] += 0.12 pos[k] = coord nx.draw_networkx_labels(G, pos, weight_of_rotation, font_size=8)
plt.axis('off') plt.show()
7 Finding the maximum weight closed subset of P’
After creating the directed graph, we added the negative node to source s and the positive node to sink t and then we applied the minimum cut by removing some rotation P’ in order to find the male optimal list.
Listing 1: G-shapely.py – adding the source and the sink to the graph
# Create flow graph where we will compute the minimum cut G_cut = nx.DiGraph() G_cut.add_node('s') G_cut.add_node('t') G_cut.add_nodes_from(G.nodes) G_cut.add_edges_from(G.edges)
for rotation in G.nodes: weight = int(weight_of_rotation[rotation])
if weight < 0: G_cut.add_edge('s', rotation, capacity=-weight) elif weight > 0: G_cut.add_edge(rotation, 't', capacity=weight)
removed_rotations = nx.minimum_cut(G_cut, 's', 't')[1][1] removed_rotations.remove('t') removed_rotations = list(removed_rotations)
print("Removed rotations:") print(removed_rotations)
Listing 2: G-shapely.py- sum up the maximum weight.
new_cost = male_optimal_cost
for rotation in removed_rotations: new_cost -= int(weight_of_rotation[rotation])
print("New cost: {}".format(new_cost))
If we select 2 from the two options above so the female will propose to male, as we wanted to find the male optimality we write this function for evaluating and testing the code to see how correct it is.
Listing 1: G-shapely.py-female propose to male.
elif x == 2: __initWomen__() stable_matching_women() matched_pairs = strip_m(tmp) matched_pairs.sort(key = lambda x: x[0]) print('Following are the stable pairs:') pp.pprint(matched_pairs) print("Total Cost calculated: {}".format(calc_cost(tmp)))
print("Women shortlist:") pp.pprint(shortlist_women(women_cp, tmp))
sw = shortlist_women(women_cp, tmp)
print("Men shortlist:") mw = shortlist_men(men_cp, sw, tmp) _mw = dict(mw)
cv = {} for k, v in sorted(_mw.items()): cv[k] = v
da = cv db = men_cp pp.pprint(reorder(da , db))
8 Evaluation and testing
We tested and evaluated the code by using the male and female lists from (Irving and leather)’s article to make sure this code is producing a correct output. See below
men_list = { '1': ['3', '1', '5', '7', '4', '2', '8', '6'], '2': ['6', '1', '3', '4', '8', '7', '5', '2'], '3': ['7', '4', '3', '6', '5', '1', '2', '8'], '4': ['5', '3', '8', '2', '6', '1', '4', '7'], '5': ['4', '1', '2', '8', '7', '3', '6', '5'], '6': ['6', '2', '5', '7', '8', '4', '3', '1'], '7': ['7', '8', '1', '6', '2', '3', '4', '5'], '8': ['2', '6', '7', '1', '8', '3', '4', '5'] }
women_list = { '1': ['4', '3', '8', '1', '2', '5', '7', '6'], '2': ['3', '7', '5', '8', '6', '4', '1', '2'], '3': ['7', '5', '8', '3', '6', '2', '1', '4'], '4': ['6', '4', '2', '7', '3', '1', '5', '8'], '5': ['8', '7', '1', '5', '6', '4', '3', '2'], '6': ['5', '4', '7', '6', '2', '8', '3', '1'], '7': ['1', '4', '5', '6', '2', '8', '3', '7'], '8': ['2', '5', '4', '3', '7', '8', '1', '6'] }
