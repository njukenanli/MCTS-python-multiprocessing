from typing import *
import gc
from queue import Queue
import heapq
from multiprocessing import Process, Manager, Lock
import json
from copy import deepcopy
import os

class NODE():
    childlist = []
    N: int = 0
    Nlist: List[int] = []
    qlist: List[float] = []
    Plist: List[float] = []
    V = 0 #Union[int, float], you can use 1, -1, or float num in (-1.0,1.0)
    #if is leaf, V = v from gen_value method, else V = G = mean(Vi for i in children_chosen)
    choice_len: int = 0
    def __init__(self, childlist, Plist: List[float], V: Optional[float]):
        self.childlist = childlist
        self.Plist = Plist
        self.V = V
        self.N = 0
        self.choice_len = len(childlist)
        self.Nlist = [0] * self.choice_len
        self.qlist = [0.0] * self.choice_len
        return
    def __str__(self):
        return "childlist: {}, Nlist: {}\n".format(self.childlist, self.Nlist)

class ROUTE():
    success: int = -1 # in {-1, 1}
    states_to_solve = [] # for next iteration
    route = dict()
    mean_prob = 1 # in [0, 1]
    route_len = 0 
    train_data = [] # in train mode, intermediate policy and value would be retained for RL use
    root = None
    def __init__(self, root):
        self.states_to_solve = [root]
        self.root = root
        self.success = -1
        self.route = dict()
        self.mean_prob = 1
        self.route_len = 0
        self.train_data = []
    def __lt__(self, other):
        return self.mean_prob >= other.mean_prob
    def __str__(self):
        return "success state: {}, probability: {}, states_to_solve: {}, route_dict: {}\n"\
                .format(self.success, self.mean_prob, self.states_to_solve, self.route)
    def gen_tree(self):
        return {self.root: self._trace(self.route, self.root)}
    def _trace(self, d, root):
        '''
        protected
        '''
        if root not in d:
            return None
        else:
            return dict([[i, self._trace(d, i)] for i in d[root]])

class MCTS_BASE():
    '''
    Users can define their own child class inherited from MCTS_BASE
    '''
    temp_coef: float = 1.0 # this is tau
    Cpuct: float = 1.0
    max_route_len: int = 15
    max_search_depth: int = 8
    mcts_times: int = 400
    update_method: str = 'avg'
    search_space = dict() # map<value of node : node> value of node is defined by user
    debug: bool = False # print process
    gen_choice: Callable # function defined by user in the child class
    #input current stautus
    # return; (value function for current status, 
    #([child_node1[branch1,2,3...],child_node2,3...],[child node probability distribution p1,p2,p3...]))
    gen_value: Optional[Callable] = None # function defined by user in the child class
    #if your method would yield v and p together, then let gen_value = None
    #else set value function for current status in the return of gen_choice to be None and 
    is_end: Callable #bool is_end(current_status)
    class PrioritizedItem():
        parent_idx: int # position in return_list[route_idx]
        idx: int # position in Plist/childlist
        value: float # probability, key
        def __init__(self, parent_idx, idx, value):
            self.parent_idx = parent_idx
            self.idx = idx
            self.value = value
        def __lt__(self, other):
            return self.value >= other.value
    def __init__ (self, gen_choice: Callable, is_end: Callable, gen_value: Optional[Callable] = None,
            temp_coef: float = 1.0, max_route_len: int = 15, max_search_depth: int = 8, mcts_times: int = 400,
            Cpuct: float = 1.0, update_method: str = 'avg', debug: bool = False):
        self.temp_coef = temp_coef
        self.Cpuct = Cpuct
        self.max_route_len = max_route_len
        self.max_search_depth = max_search_depth
        self.mcts_times = mcts_times
        self.gen_choice = gen_choice
        self.gen_value = gen_value
        self.is_end = is_end
        self.update_method = update_method
        self.debug = debug
        return
    def _gen_U(self, p: float, sumn: int, nsa: int) -> float: 
        '''
        protected: this function cannot be accessed by user, but can be inherited by child class
        Users can use their own evaluation method to rewrite these in the child class.
        '''
        return self.Cpuct * p * (sumn**0.5) / (1+nsa)
    def _gen_Q(self, oldQ: float, nsa: int, Gs) -> float: 
        '''
        protected
        Users can use their own evaluation method to rewrite these in the child class.
        '''
        return oldQ + (Gs-oldQ)/nsa
    def _gen_V(self, vlist):
        '''
        protected
        This function counts the V(some paper call it G) of a parent node 
        from all the chilren of one choice from the parent node.
        You can use 'avg' method or 'min' method, or rewrite the function in the child class.
        Other factors may also be considered in the evaluation of V in the child class:
        if so, _dfs method may also need to be rewritten...
        '''
        for i in vlist:
            if i == -1:
                return -1
        if (self.update_method == 'avg'):
            return sum(vlist)/len(vlist)
        elif (self.update_method == 'min'):
            return min(vlist)
        else:
            raise("unknoen update_method type: in the MCTS base class, only avg and min method can be chosen")
    def _argmax(self, l):
        '''
        protected
        '''
        maxvalue = l[0]
        maxindex = 0
        for i in range(1,len(l)):
            if l[i] > maxvalue:
                maxvalue = l[i]
                maxindex = i
        return maxindex
    def play(self, root, idx :int, alternatives: int = 1, train: bool = True, 
            max_train_data_num: int = 5000) -> None:
        '''
        public: This is the MCTS entrance for the user.
        return;
        shape: number of alternative routes * each route/tree expressed as generalized table
        [answer1{root: {node1@layer1: {children of node1...}, node2@layer1: {} }} , answer2 ...]
        If alternatives == k>1 the program would generate k routes with the beam search.
        I suggest letting alternatives = 1 while training for RL.
        '''
        #Step1 varify input need to be solved
        if self.is_end(root):
            with open("answers_"+str(idx), mode = "w+") as f:
                json.dump([1, {root: None}], f, indent = True)
            return
        
        #Step2 prepare to solve
        routelist = [ROUTE(root)] # for multi_route design it would be expanded
        manager = Manager() # here we use multi_process parallelism for acceleration
        lock = Lock()
        self.search_space = manager.dict(self.search_space) # make it shared across processes
        finished_route = []
        
        #Step3 solve for each iteration
        for iteration in range(self.max_route_len):
            return_list = [manager.list([]) for a in range(len(routelist))]
            processes = []
            for a in range(len(routelist)):
                processes.extend( [Process(target=self._dfs_main, args=(status, self.search_space, return_list[a], lock))\
                    for status in routelist[a].states_to_solve] ) 
            #here we put all alternative routes into parallel computation through muti-processes
            [k.start() for k in processes]
            [k.join() for k in processes]
            
            #Step4 analyse these multiple routes computed and adopt beam search
            # return_list dim : 
            #   [dim1: answer of each route;
            #   dim2: answer of each branch in this route; 
            #   dim3: (v, p, root, childlist) for this branch]
            new_route_list = [] # for next iteration
            for a in range(len(routelist)):
                #for each route 
                failed = False
                for last_state in return_list[a]:
                    if not last_state[3]:
                        failed = True
                        break
                if failed:
                    routelist[a].success = -1
                    routelist[a].mean_prob = 0
                    routelist[a].states_to_solve = []
                    new_route_list.append(routelist[a])
                    continue
                if train:
                    routelist[a].train_data.extend(return_list[a])

                #go to beam search
                routelist[a].states_to_solve = [] #insert states to solve in the next round
                beam_count = alternatives
                heaplist = [] 
                # adopt a priority queue (heap) to find top-kth for each branch
                picked_policy_list = [] # used to pick top-kth route for the next round
                for last_state_id in range(len(return_list[a])):
                    heaplist.append([self.PrioritizedItem(last_state_id, idx, return_list[a][last_state_id][1][idx])  \
                            for idx in range(self.search_space[return_list[a][last_state_id][2]].choice_len)])
                    heapq.heapify(heaplist[-1])
                newroute = deepcopy(routelist[a])
                newroute.mean_prob = newroute.mean_prob**newroute.route_len
                for last_state_id in range(len(return_list[a])):
                    item = heapq.heappop(heaplist[last_state_id])
                    newroute.route[return_list[a][last_state_id][2]] = return_list[a][last_state_id][3][item.idx]
                    branch_list = [] #save states to solve
                    for next_state in newroute.route[return_list[a][last_state_id][2]]:
                        if not self.is_end(next_state):
                            newroute.states_to_solve.append(next_state)
                            branch_list.append(next_state)
                    newroute.route_len += 1
                    if branch_list:
                        newroute.mean_prob *= return_list[a][last_state_id][1][item.idx]
                        picked_policy_list.append((return_list[a][last_state_id][1][item.idx], branch_list))
                    else:
                        picked_policy_list.append((1, branch_list))
                if not newroute.states_to_solve:
                    newroute.mean_prob = 1
                    newroute.success = 1
                else:
                    newroute.mean_prob = newroute.mean_prob**(1/newroute.route_len)
                new_route_list.append(newroute)
                beam_count -= 1
                # if it is for single path, then this while iteration will be skipped
                top_nth_heap = []
                while (beam_count > 0):
                    if not top_nth_heap:
                        for last_state_id in range(len(return_list[a])):
                            if not heaplist[last_state_id]:
                                continue
                            top_nth_heap.append(heapq.heappop(heaplist[last_state_id]))
                        if not top_nth_heap:
                            break #if there's noting to add, we cannot collect enough routes, just break
                        heapq.heapify(top_nth_heap)
                    item = heapq.heappop(top_nth_heap) 
                    # for the 2nd best choice, we substitute only 1 branch with the max 2nd largest value
                    newroute = deepcopy(new_route_list[-1])
                    newroute.route[return_list[a][item.parent_idx][2]] = return_list[a][item.parent_idx][3][item.idx]
                    branch_list = []
                    for next_state in newroute.route[return_list[a][item.parent_idx][2]]:
                        if not self.is_end(next_state):
                            branch_list.append(next_state)
                    newroute.mean_prob = newroute.mean_prob**newroute.route_len / picked_policy_list[item.parent_idx][0]
                    if branch_list:
                        newroute.mean_prob *= return_list[a][item.parent_idx][1][item.idx]
                        picked_policy_list[item.parent_idx] = (return_list[a][item.parent_idx][1][item.idx], branch_list)
                    else:
                        picked_policy_list[item.parent_idx] = (1, branch_list)
                    newroute.states_to_solve = []
                    for next_state in picked_policy_list:
                        newroute.states_to_solve.extend(next_state[1])
                    if not newroute.states_to_solve:
                        newroute.mean_prob = 1
                        newroute.success = 1
                    else: 
                        newroute.mean_prob = newroute.mean_prob**(1/newroute.route_len)
                    new_route_list.append(newroute)
                    beam_count -= 1
            beam_count = alternatives
            del routelist
            routelist = []
            heapq.heapify(new_route_list)
            while (beam_count and new_route_list):
                item = heapq.heappop(new_route_list)
                if item.success == 1:
                    finished_route.append(item)
                    alternatives -= 1
                else:
                    routelist.append(item)
                beam_count -= 1
            del new_route_list
            if (not alternatives):
                break
            
            #(Optional) print intermediate process
            if (self.debug):
                print("iteration", iteration)
                print("search_space:")
                for item in self.search_space:
                    print(item,":",self.search_space[item])
                print("routelist:")
                for item in routelist:
                    print(item)

            #Step5 garbage collection: discard branches not selected
            q = Queue(maxsize = -1) #BFS
            retained = set()
            for route in routelist:
                for s in route.states_to_solve:
                    q.put(s)
            while (not q.empty()):
                nodev = q.get()
                if nodev not in retained:
                    retained.add(nodev)
                    if nodev in self.search_space:
                        for branches in self.search_space[nodev].childlist:
                            for next_state in branches:
                                q.put(next_state)
            oldkeys = self.search_space.keys()
            for i in range(len(oldkeys)):
                if oldkeys[i] not in retained:
                    self.search_space.pop(oldkeys[i])
            del oldkeys, retained, return_list
            gc.collect()
        
        #Step6 save output route and (Optional) RL train data
        if alternatives > 0:
            finished_route.extend(routelist[0:alternatives])
        for route_idx in range(len(finished_route)):
            ans = finished_route[route_idx].gen_tree()
            with open("answers_"+str(idx)+"_route_"+str(route_idx), mode = "w+") as f:
                json.dump([finished_route[route_idx].success, ans], f, indent = True)
        train_data = []
        for route in finished_route:
            for train_data_idx in range(len(route.train_data)):
                route.train_data[train_data_idx][0] = route.success
            train_data.extend(route.train_data)
        if os.path.exists("train_data.json"):
            with open("train_data.json") as f:
                train_data.extend(json.load(f))
        with open("train_data.json", mode ="w+") as f:
            json.dump(train_data[0:max_train_data_num], f)
        return
    
    def _dfs_main(self, root, search_space_manager, return_list_manager, lock):
        '''
        protected: this function cannot be accessed by user, but can be inherited by child class
        return (value ,policy, root. chilren) of current state
        '''
        #dfs entrance
        for count in range(self.mcts_times):
            self._dfs(root, search_space_manager, 1, lock)
        #generate policy vector
        policy = [0.0]*search_space_manager[root].choice_len
        norm = 0
        for i in range(search_space_manager[root].choice_len):
            policy[i] = search_space_manager[root].Nlist[i]**(1/self.temp_coef)
        norm = sum(policy)
        for i in range(search_space_manager[root].choice_len):
            policy[i] /= norm
        return_list_manager.append([search_space_manager[root].V, policy, root, search_space_manager[root].childlist])
    
    def _dfs(self, root, search_space_manager, depth: int, lock): 
        '''
        protected
        This is the backbone of MCTS: For reference see AlphagoZero
        Note: when changing shared data, there's a need to lock the processes. 
        If we only visit shared data, there's no need to lock.
        '''
        if root not in search_space_manager:
            choices = self.gen_choice(root)
            search_space_manager[root] = NODE(choices[1][0], choices[1][1],
                    choices[0] if choices[0] is not None else None)
        lock.acquire()
        node = search_space_manager[root]
        node.N += 1
        search_space_manager[root] = node
        lock.release()
        if not search_space_manager[root].choice_len:
            node = search_space_manager[root]
            node.V = -1
            search_space_manager[root] = node
        else:
            #Select
            maxPUCT = 0.0
            maxindex = -1
            #print("Ns", search_space_manager[root].N)
            for i in range(search_space_manager[root].choice_len):
                tempPUCT = search_space_manager[root].qlist[i] + self._gen_U(search_space_manager[root].Plist[i],
                    search_space_manager[root].N, search_space_manager[root].Nlist[i])
                #print(root,search_space_manager[root].childlist[i],tempPUCT)
                if tempPUCT > maxPUCT:
                    maxPUCT = tempPUCT
                    maxindex = i
            next_state_list = search_space_manager[root].childlist[maxindex]
            #Expand and Evaluate
            rec_list = []
            vlist = []
            rec_states = []
            sumv = 0
            for next_state in next_state_list:
                if self.is_end(next_state):
                    vlist.append(1)
                elif depth == self.max_search_depth:
                    if next_state not in search_space_manager:
                        choices = self.gen_choice(next_state)
                        search_space_manager[next_state] = NODE(choices[1][0], choices[1][1],
                            choices[0] if choices[0] is not None else None)
                    if search_space_manager[next_state].V is None:
                        node = search_space_manager[next_state]
                        node.V = self.get_value(next_state)
                        search_space_manager[next_state] = node
                    if (search_space_manager[next_state].V == -1):
                        sumv = -1
                        break
                    vlist.append(search_space_manager[next_state].V)
                else:
                    rec_list.append(Process(target=self._dfs, args=(next_state, search_space_manager, depth+1, lock)))
                    rec_states.append(next_state)
            if sumv != -1:
                [k.start() for k in rec_list]
                [k.join() for k in rec_list]
                for next_state in rec_states:
                    vlist.append(search_space_manager[next_state].V)
                sumv = self._gen_V(vlist)
            #Backup
            lock.acquire()
            node = search_space_manager[root]
            node.V = sumv
            node.Nlist[maxindex] += 1
            node.qlist[maxindex] = self._gen_Q(node.qlist[maxindex], node.Nlist[maxindex], sumv)
            search_space_manager[root] = node
            lock.release()
            #print(root,next_state_list, search_space_manager[root].Nlist)
        return
