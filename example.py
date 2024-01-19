from mcts_base import *
import json
class MY_MCTS(MCTS_BASE):
    choices = dict() # I suggest using dict and set for find operation 
    end_set = set() # because in python dict and set are based on Hash table so find operation takes 0(1)
    def __init__(self):
        with open("example.json") as f:
            l = json.load(f)
        self.choices = l["choice"]
        self.end_set = set(l["is_end"])
        super().__init__(gen_choice = self.gen_choice, is_end = self.is_end, gen_value = None,
                mcts_times = 3, max_route_len = 6, max_search_depth = 2, debug = True)
    def gen_choice(self, status):
        if status not in self.choices:
            return [-1, [[],[]]]
        else:
            return self.choices[status]
    def is_end(self, status):
        return (status in self.end_set)

if __name__ == '__main__':
    print("below is mcts.play(\"a\", 0, alternatives = 1, train = True)")
    print("single best route planning. MCTS data generated can be used for Reinforcement Lrearning.")
    mcts = MY_MCTS()
    mcts.play("a", 0)
    del mcts
    print("below is mcts.play(\"a\", 1, alternatives = 3, train = False)")
    print("muliple/alternative routes planning")
    mcts = MY_MCTS()
    mcts.play("a", 1, alternatives = 3, train = False)

'''
To run it: python example.py > log.out
'''
