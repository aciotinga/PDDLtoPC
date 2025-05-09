import pddlgym
from pddlgym.structs import Literal
from pddlgym.structs import Predicate

from typing import List, Set, Dict, Tuple

# Define the PC nodes
class AbstractNode:
    def __init__(self, node_type: str, scope: Set[int], children: List):
        self.node_type = node_type
        self.scope = scope
        self.children = children
    
    def copy(self):
        raise NotImplementedError(f"Copy not implemented for {self.node_type} nodes!")

    def forward(self, data: List[float]):
        raise NotImplementedError(f"Forward not implemented for {self.node_type} nodes!")

    def marginalize(self, vars: Set[int]):
        raise NotImplementedError(f"Marginalize not implemented for {self.node_type} nodes!")

    def normalize(self):
        raise NotImplementedError(f"Normalize not implemented for {self.node_type} nodes!")
        
    def size(self):
        raise NotImplementedError(f"Size not implemented for {self.node_type} nodes!")
        
    def map(self):
        raise NotImplementedError(f"MAP not implemented for {self.node_type} nodes!")

class SumNode(AbstractNode):
    def __init__(self, scope: Set[int], params: List[float], children: List[AbstractNode]):
        super().__init__("sum", scope, children)

        self.params = params

    def forward(self, data: List[float]):
        total = 0
        for i in range(len(self.params)):
            total += self.params[i] * self.children[i].forward(data)
        return total
        
    def copy(self):
        copies = []
        for c in self.children:
            copies.append(c.copy())
        return SumNode(self.scope.copy(), self.params.copy(), copies)
        
    def size(self):
        edge_count = 0
        for c in self.children:
            edge_count += c.size() + 1
        return edge_count
        
    def map(self):
        max_prob = 0
        max_assignment = None
        for i, c in enumerate(self.children):
            p, a = c.map()
                
            # Greater than max
            if p * self.params[i] > max_prob:
                max_prob = p * self.params[i]
                max_assignment = a
                
        return (max_prob, max_assignment)
        

class ProductNode(AbstractNode):
    def __init__(self, scope: Set[int], children: List[AbstractNode]):
        super().__init__("product", scope, children)

    def forward(self, data: List[float]):
        total = 1
        for i in range(len(self.children)):
            total *= self.children[i].forward(data)
        return total
        
    def copy(self):
        copies = []
        for c in self.children:
            copies.append(c.copy())
        return ProductNode(self.scope.copy(), copies)
        
    def size(self):
        edge_count = 0
        for c in self.children:
            edge_count += 1 + c.size()
        return edge_count
        
    def map(self):
        max_prob = 1
        max_assignment = {}
        for i, c in enumerate(self.children):
            p, a = c.map()
                
            max_prob *= p
            for k in a.keys():
                max_assignment[k] = a[k]
                
        return (max_prob, max_assignment)

class CategoricalInputNode(AbstractNode):
    def __init__(self, scope: Set[int], params: List[float]):
        super().__init__("input", scope, [])

        self.params = params

    def forward(self, data: List[int]):
        data_val = next(iter(self.scope))

        # If not in the data, we marginalize this node in the forward pass
        return self.params[data[data_val]] if data_val in data.keys() else 1
        
    def copy(self):
        return CategoricalInputNode(self.scope.copy(), self.params.copy())

    def size(self):
        return 0
        
    def map(self):
        max_prob = max(self.params)
        max_assignment = {next(iter(self.scope)): self.params.index(max_prob)}
                
        return (max_prob, max_assignment)

from typing import Dict, List

def initial_state_pc(state: Dict[pddlgym.structs.Literal, bool], preds_to_vars: Dict[str, int]):
    children = []
    for pred in preds_to_vars.keys():
        if pred in state.keys(): # If explicitly mentioned, the predicate is whatever the assignment is; otherwise, it is False
            children.append(CategoricalInputNode(set([preds_to_vars[pred]]), [0, 1] if state[pred] else [1, 0]))
        else:
            children.append(CategoricalInputNode(set([preds_to_vars[pred]]), [1, 0]))
    root = ProductNode(set.union(*[c.scope for c in children]), children)
    return root

def update_pc_with_action(pc: AbstractNode, precondition, effect, preds_to_vars: Dict[str, int]):
    # All no_change clauses map to here
    no_change_pc = pc.copy()

    # We assume that the effect is a ProbabilisticEffect over LiteralConjunctions
    assert type(effect) == pddlgym.structs.ProbabilisticEffect, "Effect must be a ProbabilisticEffect!"
    assert type(precondition) == pddlgym.structs.LiteralConjunction, "Precondition must be a LiteralConjunction!"

    # Firstly, if the precondition does not hold ever, we simply return no change
    if not precondition_holds(pc, precondition, preds_to_vars):
        return no_change_pc
    
    # Thus, we create a circuit for each of the LiteralConjunctions and disjunct them with weights given by the ProbabilisticEffect
    weights = []
    subcircuits = []
    for i, lc in enumerate(effect.literals):
        # Ignore 0-probability outcomes
        if effect.probabilities[i] == 0:
            continue
            
        assert type(lc) == pddlgym.structs.LiteralConjunction, "ProbabilisticEffect must be over LiteralConjunctions!"

        # The weight for the sub-circuit we are about to make
        weights.append(effect.probabilities[i])

        # Check whether this is a NoChange clause; if so, we handle this separately and continue
        if len(lc.literals) == 1 and lc.literals[0].pddl_str() == '(NOCHANGE)':
            subcircuits.append(no_change_pc)
            continue

        # Not a no-change; we want to marginalize out from the new PC and create a new conjunction based on the effect
        variables_to_update = {}
        for l in lc.literals:
            assert type(l) == pddlgym.structs.Literal, "ProbabilisticEffect LiteralConjunctions must be single literals!"
            
            literal_var = l if not l.is_anti else l.inverted_anti
            circuit_var = preds_to_vars[literal_var.pddl_str()]
            
            variables_to_update[circuit_var] = [0, 1] if not l.is_anti else [1, 0]

        '''# For each variable to update, marginalize from old PC and conjunct it with the updated distribution
        updated_input_nodes = [CategoricalInputNode({v}, variables_to_update[v]) for v in variables_to_update.keys()]
        
        marginalized_pc = pc.copy()
        marginalized_pc.marginalize({v for v in variables_to_update.keys()})

        # New subcircuit
        new_subcircuit = ProductNode(pc.scope.copy(), [marginalized_pc] + updated_input_nodes)'''

        # For each variable to update, assign its parameters in the new pc
        new_subcircuit = pc.copy()
        update_leaf_weights(new_subcircuit, variables_to_update)
        
        subcircuits.append(new_subcircuit)

    # Now that we have weights and subcircuits, create the effect PC
    normalized_weights = [w / sum(weights) for w in weights]
    effect_pc = SumNode(pc.scope.copy(), normalized_weights, subcircuits)

    # We know that preconditions are simply a LiteralConjunction, so we just need to compute a single forward pass
    # Build out query here
    query = {}
    for l in precondition.literals:
        assert type(l) == pddlgym.structs.Literal, "Precondition must be a LiteralConjunction of Literals!"
        literal_var = l if not l.is_negative else l.positive
        circuit_var = preds_to_vars[literal_var.pddl_str()]

        value = 1 if not l.is_negative else 0

        query[circuit_var] = value
    
    # Probability that precondition holds
    pr_precond = pc.forward(query)
    
    # Lastly, our new state distribution PC is a sum over the effect_pc with weight pr_precond and no_change_pc with weight 1-pr_precond
    state_distribution = SumNode(pc.scope.copy(), [pr_precond, 1 - pr_precond], [effect_pc, no_change_pc])

    return state_distribution

def update_leaf_weights(pc: AbstractNode, weights_to_change: Dict[int, List[float]]):
    if pc.node_type == "input":
        var = next(iter(pc.scope))
        if var in weights_to_change.keys():
            pc.params = weights_to_change[var]
    else:
        for c in pc.children:
            update_leaf_weights(c, weights_to_change)

import numpy as np, gurobipy
from gurobipy import GRB

def solve_transportation_problem_new(costs, supply, demand, model):
    # Create model
    model.remove(model.getConstrs())  # Remove all constraints
    model.remove(model.getVars())    # Remove all variables
    model.reset()                    # Reset model state
    
    # Decision variables: x[i,j] = quantity shipped from source i to destination j
    x = {}
    for i in range(len(supply)):
        for j in range(len(demand)):
            x[i,j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x[{i},{j}]", lb=0)
    
    # Objective: Minimize total transportation cost
    model.setObjective(
        sum(costs[i][j] * x[i,j] for i in range(len(supply)) for j in range(len(demand))),
        GRB.MINIMIZE
    )
    
    # Supply constraints: Outflow from each source <= supply
    for i in range(len(supply)):
        model.addConstr(
            sum(x[i,j] for j in range(len(demand))) <= supply[i],
            name=f"Supply_{i}"
        )
    
    # Demand constraints: Inflow to each destination >= demand
    for j in range(len(demand)):
        model.addConstr(
            sum(x[i,j] for i in range(len(supply))) >= demand[j],
            name=f"Demand_{j}"
        )
    
    # Optimize
    model.optimize()

    v = model.objVal
    return v

def cw_dist(pc1, pc2, model=None):
    if model is None:
        model = gurobipy.Model("Transportation")
        model.setParam('OutputFlag', 0)

    # If pc1 and pc2 are sums
    if pc1.node_type == pc2.node_type and pc1.node_type == 'sum':
        # Trivially, if both sum nodes have 1 child:
        if len(pc1.children) == len(pc2.children) and len(pc1.children) == 1:
            return cw_dist(pc1.children[0], pc2.children[0], model=model)
        if len(pc2.children) == 1:
            cw_val = 0
            for i, c1 in enumerate(pc1.children):
                cw_val += pc1.params[i] * cw_dist(c1, pc2.children[0], model=model)
            
            return cw_val
        if len(pc1.children) == 1:
            cw_val = 0
            for j, c2 in enumerate(pc2.children):
                cw_val += pc2.params[j] * cw_dist(pc1.children[0], c2, model=model)
            
            return cw_val

        # Transport costs
        costs = []
        for i, c1 in enumerate(pc1.children):
            r = []
            for j, c2 in enumerate(pc2.children):
                r.append(cw_dist(c1, c2, model=model))
            costs.append(r)
        supply = pc1.params
        demand = pc2.params

        result = solve_transportation_problem_new(costs, supply, demand, model=model)

        return result

    # If pc1 and pc2 are products
    if pc1.node_type == pc2.node_type and pc1.node_type == 'product':
        cw_val = 0
        for i, c1 in enumerate(pc1.children):
            for j, c2 in enumerate(pc2.children):
                if c1.scope != c2.scope:
                    continue
                cw_val += cw_dist(c1, c2, model=model)
        
        return cw_val

    # If pc1 and pc2 are inputs
    if pc1.node_type == pc2.node_type and pc1.node_type == 'input':
        return abs(pc1.params[0] - pc2.params[0])

    # Here, we know that either pc1 or pc2 is a sum node, while the other is either a product or input node.
    if pc1.node_type == 'sum':
        cw_val = 0
        for i, c1 in enumerate(pc1.children):
            cw_val += pc1.params[i] * cw_dist(c1, pc2, model=model)
        
        return cw_val
        
    if pc2.node_type == 'sum':
        cw_val = 0
        for j, c2 in enumerate(pc2.children):
            cw_val += pc2.params[j] * cw_dist(pc1, c2, model=model)
        
        return cw_val

def approximate_tv(pc1, pc2, assignments={}):
    best_mar_dist = 0
    best_mar_var = None
    # We first look at each candidate variable we can add and evaluate what the tv distance does
    for var in pc1.scope:
        if var in assignments.keys():
            continue

        # First, try var=0 and then var=1
        new_assignment = assignments.copy()
        
        new_assignment[var] = 0

        p_x = pc1.forward(new_assignment)
        p_y = pc2.forward(new_assignment)

        if abs(p_x-p_y) >= best_mar_dist:
            best_mar_dist = abs(p_x-p_y)
            best_mar_var = (var, 0)
            
        new_assignment[var] = 1

        p_x = pc1.forward(new_assignment)
        p_y = pc2.forward(new_assignment)

        if abs(p_x-p_y) >= best_mar_dist:
            best_mar_dist = abs(p_x-p_y)
            best_mar_var = (var, 1)
    
    if best_mar_var is None:
        return None

    # Create a new assignment to build off
    assignments[best_mar_var[0]] = best_mar_var[1]
    
    sub_result = approximate_tv(pc1, pc2, assignments=assignments.copy())
    
    if sub_result is None:
        return best_mar_dist, assignments
    else:
        return sub_result[0], sub_result[1]

# Determines whether a given precondition can possibly hold for a state distribution
def precondition_holds(pc, precondition, predicates_to_vars):
    query = {}
    for l in precondition.literals:
        assert type(l) == pddlgym.structs.Literal, "Precondition must be a LiteralConjunction of Literals!"
        literal_var = l if not l.is_negative else l.positive
        circuit_var = predicates_to_vars[literal_var.pddl_str()]
    
        value = 1 if not l.is_negative else 0
    
        query[circuit_var] = value

    return pc.forward(query) > 0