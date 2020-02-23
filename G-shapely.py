import collections
import json
import matplotlib.pyplot as plt
import networkx as nx
import pprint

from copy import deepcopy

pp = pprint.PrettyPrinter() # Pretty printer

def m(mydict):
    """This function appends the character 'm' to the keys
    of the male preference list and 'w' to all the items
    in the value.

    Parameters
    ----------
    mydict: A dict. The male preference list

    Returns
    ----------
    A modified dict.

    """

    return {k + 'm': [i + 'w' for i in v] for k, v in mydict.items()}


def f(mydict):
    """This function appends the character 'w' to the keys
    of the female preference list and 'm' to all the items
    in the value.

    Parameters
    ----------
    mydict: A dict. The female preference list

    Returns
    ----------
    A modified dict.
    """

    return {k + 'w': [i + 'm' for i in v] for k, v in mydict.items()}


def strip_m(mylist):
    """A utility function to strip the unwanted characters
    from the resulting stable pairs.

    Parameters
    ----------
    mylist: A list. All the resulting stable pairs.

    Returns
    ----------
    The required stable pairs.
    """

    z = [item for sublist in mylist for item in sublist]

    p = []
    for i in range(0, len(z) - 1, 2):
        p.append([z[i][:-1], z[i + 1][:-1]])
    return p


tmp = []
free_men = []
free_women = []


def __initMen__():
    """We initialise the method so we can pair every
    men from the given preference list.
    """

    for i in _men.keys():
        free_men.append(i)

def __initWomen__():
    """We initialise the method so we can pair every
    women from the given preference list.
    """
    for i in _women.keys():
        free_women.append(i)


def men_propose(man):
    """This method will be run when we need men to propose first
    to the woman according to preference list.

    Parameters
    ----------
    man: An int. This represents the man in the free_men list.
    """

    for woman in _men[man]:

        # Boolean for whether a woman is taken or not
        taken_match = [couple for couple in tmp if woman in couple]

        if (len(taken_match) == 0):
            # Engage the man and  the woman
            tmp.append([man, woman])
            free_men.remove(man)
            break

        elif (len(taken_match) > 0):
            # Check ranking of the current male and the ranking of the 'to-be' male
            current_guy = _women[woman].index(taken_match[0][0])
            potential_guy = _women[woman].index(man)

            if (current_guy < potential_guy):
                z = taken_match[0][0]
            else:
                # The new guy is engaged
                free_men.remove(man)

                # The previous male is now single
                free_men.append(taken_match[0][0])

                # Update the fiance of the woman (tentatively)
                taken_match[0][0] = man
                break

def women_propose(woman):
    """This method will be run when we need women to propose first
    to the man according to preference list.

    Parameters
    ----------
    woman: An int. This represents the woman in the free_women list.
    """
    for man in _women[woman]:

        # Boolean for whether a woman is taken or not
        taken_match = [couple for couple in tmp if man in couple]

        if (len(taken_match) == 0):
            # Engage the man and  the woman
            tmp.append([woman, man])
            free_women.remove(woman)
            break

        elif (len(taken_match) > 0):
            # Check ranking of the current female and the ranking of the 'to-be' female
            current_woman = _men[man].index(taken_match[0][0])
            potential_woman = _men[man].index(woman)

            if (current_woman < potential_woman):
                z = taken_match[0][0]
            else:
                # The new woman is engaged
                free_women.remove(woman)

                # The previous female is now single
                free_women.append(taken_match[0][0])

                # Update the better half of the man (tentatively)
                taken_match[0][0] = woman
                break


def stable_matching_men():
    """
    Function to run the algorithm when men propose first until
    the pairs are matched.
    """

    while (len(free_men) > 0):
        for i in free_men:
            men_propose(i)

def stable_matching_women():
    """
    Function to run the algorithm when women propose first until
    the pairs are matched.
    """

    while (len(free_women) > 0):
        for i in free_women:
            women_propose(i)


def men_list(_men_):
    """
    We need to use the list values from the male preference dict.
    This function enlist them into a single list.

    Parameters
    ----------
    _men_: A dict. Male preference dict

    Returns
    ----------
    A list of lists
    """

    d = []
    for k, v in _men_.items():
        d.append(v)

    return d



def women_list(_women_):
    """
    We need to use the list values from the female preference dict.
    This function enlist them into a single list.

    Parameters
    ----------
    _women_: A dict. Female preference dict

    Returns
    ----------
    A list of lists
    """

    d = []
    for k, v in _women_.items():
        d.append(v)

    return d


def calc_cost(tmp):
    """
    This method calculates the total cost that is
    incorporated into building male and female optimal
    pairs.

    Parameters
    ----------
    tmp: A list. This flist contains the male and female optimal solution
        in an unordered way.

    Returns
    ----------
    total_cost: An int. This is the total cost calculated as per the cost
        function
    """

    y = []
    for i in range(len(_men)):
        y.append(men_list(_men)[i])

    pairs = strip_m(tmp)
    pairs.sort(key = lambda x: x[0])


    men_list_numbered = []

    for s in y:
        s = [w.replace('w', '') for w in s]
        men_list_numbered.append(s)


    cost_i = []
    for i in range(len(pairs)):
        cost_i.append(men_list_numbered[i].index(pairs[i][1]) + 1)



    z = []
    for i in range(len(_women)):
        z.append(women_list(_women)[i])

    pairs_for_women = strip_m(tmp)
    pairs_for_women.sort(key = lambda x: x[1])

    women_list_numbered = []

    for s in z:
        s = [w.replace('m', '') for w in s]
        women_list_numbered.append(s)


    cost_j = []
    for i in range(len(pairs_for_women)):
        cost_j.append(women_list_numbered[i].index(pairs_for_women[i][0]) + 1)



    cost_list = cost_i + cost_j
    total_cost = sum(cost_list)


    return total_cost


def shortlist_women(women_cp, tmp):
    """
    Creates shortlists for women preference list.

    Parameters
    ----------
    women_cp:  A dict. Copy of original women preference list.
    tmp:       A list. A list of optimal matchings.

    Returns
    ----------
    women_shortlist: A dict. Final shortlist for women.
    """

    strip_m(tmp)
    tmp.sort(key = lambda x: x[0])

    women_shortlist = {}

    i = 0
    for k, v in women_cp.items():
        x = tmp[i][1][:-1]
        women_shortlist[x] = (women_cp.get(x))[:(women_cp.get(x)).index(tmp[i][0][:-1]) + 1]

        i += 1

    return women_shortlist


def shortlist_men(men_cp, sw, tmp):
    """
    Creates shortlists for women preference list.

    Parameters
    ----------
    men_cp:  A dict. Copy of original women preference list.
    sw:      A dict. Women shortlist.
    tmp:     A list. A list of optimal matchings.

    Returns
    ----------
    men_shortlist: A dict. Final shortlist for men.
    """

    strip_m(tmp)
    tmp.sort(key = lambda x: x[0])

    men_shortlist = collections.defaultdict(list)
    women_list = []

    

    for k, v in sw.items():
        for j in v:
            men_shortlist[j].append(k)

    return men_shortlist

def reorder(da, db):
    """This method restores the order of preferences
    in any ranking list

    Parameters
    ----------
    da: A dict. Unordered shortlist.
    db: A dict. Original preference list
    """

    n = len(da)
    check = [0] * (n + 1)
    final = collections.defaultdict(list)

    for k, v in da.items():
        for j in range(n + 1):
            check[int(j)] = 0
        for i in v:
            check[int(i)] = 1
        for j in db[k]:
            if check[int(j)] == 1:
                final[k].append(j)
    return dict(final)


def main():
    print('1. Should Men propose first?')
    print('2. Should Women propose first?')
    print('Enter your choice(1/2)')

    x = int(input())

    if x == 1:
        __initMen__()
        stable_matching_men()
        matched_pairs = strip_m(tmp)
        matched_pairs.sort(key = lambda x: x[0])
        print('Following are the stable pairs:')
        pp.pprint(matched_pairs)

        male_optimal_cost = calc_cost(tmp)
        print("Total Cost calculated: {}".format(male_optimal_cost))

        print("Women shortlist:")
        sw = shortlist_women(women_cp, tmp)
        pp.pprint(sw)

        print("Men shortlist:")
        mw = shortlist_men(men_cp, sw, tmp)
        _mw = dict(mw)

        cv = {}
        for k, v in sorted(_mw.items()):
            cv[k] = v

        da = cv
        db = men_cp
        ordered_mw = reorder(da , db)
        pp.pprint(ordered_mw)

        # auxiliary variables for finding rotations
        rotations = []
        last_rotation_for_man = {}
        weight_of_rotation = {}
        level_of_rotation = {}
        level = 0
        ordered_mw_copy = ordered_mw.copy()

        G = nx.DiGraph()
        labels = {}

        # Loop that goes throught the levels of rotations until it doesn't
        # find any more rotations
        while True:
            # Go through the unvisited nodes and find cycles
            exposed_rotations = []
            visited = set()
            for k in ordered_mw.keys():
                if not k in visited:
                    current = k
                    rotation = []
                    while current not in visited:
                        visited.add(current)
                        # A man with only one woman in his shortlist was found
                        # Hence we won't be able to find any rotation in this path
                        if len(ordered_mw[current]) < 2:
                            rotation.clear()
                            break
                        # add a tuple with (man, current partner, next partner)
                        rotation.append((current, ordered_mw[current][0], ordered_mw[current][1]))
                        # move to next man
                        current = sw[ ordered_mw[current][1] ][-1]

                    while len(rotation) > 0 and rotation[0][0] != current:
                        rotation.pop(0)

                    # add to list of rotations if a rotation was found
                    if len(rotation) > 0:
                        exposed_rotations.append(rotation)

            # there aren't any more rotations
            if len(exposed_rotations) == 0:
                break

            # Loop that eliminates the rotations and calculates the weight of
            # each rotation
            for rotation in exposed_rotations:
                rotations.append(rotation)
                current_rotation = len(rotations)
                level_of_rotation[current_rotation] = level
                weight = 0

                G.add_node(current_rotation)
                labels[current_rotation] = 'p{}'.format(current_rotation)
                

                for man, woman, new_woman in rotation:
                    weight += men_cp[man].index(woman) - men_cp[man].index(new_woman)
                    weight += women_cp[woman].index(man)
                    weight -= women_cp[new_woman].index(man)

                    if man in last_rotation_for_man:
                        predecessor = last_rotation_for_man[man]
                        G.add_edge(predecessor, current_rotation)

                    last_rotation_for_man[man] = current_rotation

                    if ordered_mw[man][0] == woman:
                        ordered_mw[man].pop(0)

                    sw[new_woman].pop(-1)
                    while sw[new_woman][-1] != man:
                        x = sw[new_woman][-1]
                        sw[new_woman].pop(-1)
                        # TODO: check
                        if new_woman in ordered_mw[x]:
                            ordered_mw[x].remove(new_woman)

                weight_of_rotation[current_rotation] = str(weight)

            level += 1

       
        # store list of edges in the graph before starting to delete some of them
        graph_edges = list(G.edges())

        # Loop to delete unnecessary edges in the graph
        # we try to remove edges to see if removing them disconnects the nodes
        # that they're connecting
        for e in graph_edges:
            G.remove_edge(e[0], e[1])

            if not nx.has_path(G, e[0], e[1]):
                G.add_edge(e[0], e[1])

        # Debug code to print the rotations and the weights
        for i, rotation in enumerate(rotations):
            rotation_str = ""
            for man, woman, new_woman in rotation:
                rotation_str += "({}, {}),".format(man, woman)
            rotation_str = rotation_str[:-1]

            print("p{} = {}".format(i + 1, rotation_str))
            print("weight = {}".format(weight_of_rotation[i + 1]))

        # Create flow graph where we will compute the minimum cut
        G_cut = nx.DiGraph()
        G_cut.add_node('s')
        G_cut.add_node('t')
        G_cut.add_nodes_from(G.nodes)
        G_cut.add_edges_from(G.edges)

        for rotation in G.nodes:
            weight = int(weight_of_rotation[rotation])

            if weight < 0:
                G_cut.add_edge('s', rotation, capacity=-weight)
            elif weight > 0:
                G_cut.add_edge(rotation, 't', capacity=weight)

        

        removed_rotations = nx.minimum_cut(G_cut, 's', 't')[1][1]
        removed_rotations.remove('t')
        removed_rotations = list(removed_rotations)

        print("Removed rotations:")
        print(removed_rotations)

        new_cost = male_optimal_cost

        for rotation in removed_rotations:
            new_cost -= int(weight_of_rotation[rotation])

        print("New cost: {}".format(new_cost))

        # Approximate distances that will be used to plot the graph
        # By default if there is not a path between nodes the distance will be
        # the difference in their levels
        dist_dict = {}
        for i in range(1, 1 + len(rotations)):
            dist_dict[i] = {}
            for j in range(1, 1 + len(rotations)):
                dist_dict[i][j] = 1 + abs(level_of_rotation[i] - level_of_rotation[j])

        # Compute shortest path and store in distance matrix
        for row, data in nx.shortest_path_length(G):
            for col, dist in data.items():
                dist_dict[row][col] = dist
                dist_dict[col][row] = dist

        # Set up the layout for plotting the graph
        pos = nx.kamada_kawai_layout(G, dist=dist_dict)
        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=removed_rotations, node_color='b', node_size=500, alpha=0.4)
        nx.draw_networkx_edges(G, pos, node_size=500)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Increase the y coordinate a bit to plot the weights of nodes
        # above the nodes
        for k, v in pos.items():
            coord = v
            coord[1] += 0.12
            pos[k] = coord
        nx.draw_networkx_labels(G, pos, weight_of_rotation, font_size=8)

        plt.axis('off')
        plt.show()

    elif x == 2:
        __initWomen__()
        stable_matching_women()
        matched_pairs = strip_m(tmp)
        matched_pairs.sort(key = lambda x: x[0])
        print('Following are the stable pairs:')
        pp.pprint(matched_pairs)
        print("Total Cost calculated: {}".format(calc_cost(tmp)))

        print("Women shortlist:")
        pp.pprint(shortlist_women(women_cp, tmp))

        sw = shortlist_women(women_cp, tmp)

        print("Men shortlist:")
        mw = shortlist_men(men_cp, sw, tmp)
        _mw = dict(mw)

        cv = {}
        for k, v in sorted(_mw.items()):
            cv[k] = v

        da = cv
        db = men_cp
        pp.pprint(reorder(da , db))


if __name__=='__main__':
    global _men, _women, men_cp, women_cp

    f_men = open('men_list.txt')
    _men = json.load(f_men)
    f_women = open('women_list.txt')
    _women = json.load(f_women)

    #Create copies of preference list for further operations
    men_cp = deepcopy(_men)
    women_cp = deepcopy(_women)

    # Append characters to male preference list
    _men = m(_men)

    # Append characters to male preference list
    _women = f(_women)

    main()