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

        self.cache = None

    def clear_cache(self):
        self.cache = None
        for c in self.children:
            c.clear_cache()

    def forward(self, data: List[float]) -> float:
        self.clear_cache()
        self._forward(data)
        return self.cache

    def normalize(self):
        self.clear_cache()
        self._normalize()

    def marginalize(self, vars: Set[int]):
        self.clear_cache()
        self._marginalize(vars)
        
    def copy(self):
        self.clear_cache()
        self._copy()
        return self.cache
        
    def size(self):
        self.clear_cache()
        self._size()
        return self.cache
        
    def _copy(self):
        raise NotImplementedError(f"Copy not implemented for {self.node_type} nodes!")

    def _forward(self, data: List[float]):
        raise NotImplementedError(f"Forward not implemented for {self.node_type} nodes!")

    def _marginalize(self, vars: Set[int]):
        raise NotImplementedError(f"Marginalize not implemented for {self.node_type} nodes!")

    def _normalize(self):
        raise NotImplementedError(f"Normalize not implemented for {self.node_type} nodes!")
        
    def _size(self):
        raise NotImplementedError(f"Size not implemented for {self.node_type} nodes!")

class SumNode(AbstractNode):
    def __init__(self, scope: Set[int], params: List[float], children: List[AbstractNode]):
        super().__init__("sum", scope, children)

        self.params = params

    def _forward(self, data: List[float]):
        if self.cache is not None:
            return
        
        total = 0
        for i in range(len(self.params)):
            self.children[i].forward(data)
            total += self.params[i] * self.children[i].cache
        self.cache = total

    def _marginalize(self, vars: Set[int]):
        if self.cache is not None:
            return

        # Check if the current node's scope is a subset of vars; if it is, then this node gets marginalized
        if self.scope.issubset(vars):
            self.cache = True

            # Delete all the children, since they are marginalized
            for c in self.children:
                del c

            return
        else:
            # Remove marginalized variables from scope
            # Note that we assume smoothness, decomposability, and normalized parameters
            for v in vars:
                if v in self.scope:
                    self.scope.remove(v)
            self.cache = False

        # Marginalize the children
        for c in self.children:
            c._marginalize(vars)
        
    def _copy(self):
        if self.cache is not None:
            return
            
        for c in self.children:
            c._copy()
        self.cache = SumNode(self.scope.copy(), self.params.copy(), [c.cache for c in self.children])
        
    def _size(self):
        edge_count = 0
        for c in self.children:
            edge_count += 1

            if c.cache is not None:
                continue
            c._size()

            edge_count += c.cache
        self.cache = edge_count
        

class ProductNode(AbstractNode):
    def __init__(self, scope: Set[int], children: List[AbstractNode]):
        super().__init__("product", scope, children)

    def _forward(self, data: List[float]):
        if self.cache is not None:
            return
            
        total = 1
        for i in range(len(self.children)):
            self.children[i].forward(data)
            total *= self.children[i].cache
        self.cache = total

    def _marginalize(self, vars: Set[int]):
        if self.cache is not None:
            return

        # Check if the current node's scope is a subset of vars; if it is, then this node gets marginalized
        if self.scope.issubset(vars):
            self.cache = True

            # Delete all the children, since they are marginalized
            for c in self.children:
                del c

            return
        else:
            # Remove marginalized variables from scope
            # Note that we assume smoothness, decomposability, and normalized parameters
            for v in vars:
                if v in self.scope:
                    self.scope.remove(v)
            self.cache = False

        # Marginalize the children
        to_remove = []
        for i, c in enumerate(self.children):
            c._marginalize(vars)

            if c.cache: # All variables have been marginalized in this child
                to_remove.append(i)

        # Remove all marginalized children from the child list
        for i in reversed(to_remove):
            del self.children[i]
        
    def _copy(self):
        if self.cache is not None:
            return
            
        for c in self.children:
            c._copy()
        self.cache = ProductNode(self.scope.copy(), [c.cache for c in self.children])
        
    def _size(self):
        edge_count = 0
        for c in self.children:
            edge_count += 1

            if c.cache is not None:
                continue
            c._size()

            edge_count += c.cache
        self.cache = edge_count

class CategoricalInputNode(AbstractNode):
    def __init__(self, scope: Set[int], params: List[float]):
        super().__init__("input", scope, [])

        self.params = params

    def _forward(self, data: List[int]):
        if self.cache is not None:
            return
        
        data_val = next(iter(self.scope))

        # If not in the data, we marginalize this node in the forward pass
        self.cache = self.params[data[data_val]] if data_val in data.keys() else 1

    def _marginalize(self, vars: Set[int]):
        if self.cache is not None:
            return

        # Check if the current node's scope is a subset of vars; if it is, then this node gets marginalized
        if self.scope.issubset(vars):
            self.cache = True
        
    def _copy(self):
        if self.cache is not None:
            return
            
        self.cache = CategoricalInputNode(self.scope.copy(), self.params.copy())

    def _size(self):
        self.cache = 0

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
        new_subcircuit.clear_cache()
        
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
    if pc.cache is not None:
        return
    pc.cache = True
    if pc.node_type == "input":
        var = next(iter(pc.scope))
        if var in weights_to_change.keys():
            pc.params = weights_to_change[var]
    else:
        for c in pc.children:
            update_leaf_weights(c, weights_to_change)

import numpy as np
import scipy.optimize

def cw_dist(pc1, pc2, memo=None):
    if memo is None:
        memo = {}
    if (pc1, pc2) in memo.keys():
        return memo[(pc1, pc2)]

    # If pc1 and pc2 are sums
    if pc1.node_type == pc2.node_type and pc1.node_type == 'sum':
        # Trivially, if both sum nodes have 1 child:
        if len(pc1.children) == len(pc2.children) and len(pc1.children) == 1:
            memo[(pc1, pc2)] = cw_dist(pc1.children[0], pc2.children[0], memo=memo)
            return memo[(pc1, pc2)]
        if len(pc2.children) == 1:
            cw_val = 0
            for i, c1 in enumerate(pc1.children):
                cw_val += pc1.params[i] * cw_dist(c1, pc2.children[0], memo=memo)
            
            memo[(pc1, pc2)] = cw_val
            return memo[(pc1, pc2)]
        if len(pc1.children) == 1:
            cw_val = 0
            for j, c2 in enumerate(pc2.children):
                cw_val += pc2.params[j] * cw_dist(pc1.children[0], c2, memo=memo)
            
            memo[(pc1, pc2)] = cw_val
            return memo[(pc1, pc2)]
            
        # Build out the LOP to solve
        A = np.zeros((len(pc1.children) + len(pc2.children), len(pc1.children) * len(pc2.children)))
        b = np.zeros(len(pc1.children) + len(pc2.children))
        c = np.zeros(len(pc1.children) * len(pc2.children))

        # Objective function
        for i, c1 in enumerate(pc1.children):
            for j, c2 in enumerate(pc2.children):
                c[i * len(pc2.children) + j] = cw_dist(c1, c2, memo=memo)

        # Build out the constraints
        c_idx = 0
        for i, c1 in enumerate(pc1.children):
            b[c_idx] = pc1.params[i]
            for j, c2 in enumerate(pc2.children):
                A[c_idx, i * len(pc2.children) + j] = 1
            c_idx += 1

        # Build out the constraints
        for j, c2 in enumerate(pc2.children):
            b[c_idx] = pc2.params[j]
            for i, c1 in enumerate(pc1.children):
                A[c_idx, i * len(pc2.children) + j] = 1
            c_idx += 1

        # We have the solution
        result = scipy.optimize.linprog(c, A_eq=A, b_eq=b)

        memo[(pc1, pc2)] = result.fun
        return memo[(pc1, pc2)]

    # If pc1 and pc2 are products
    if pc1.node_type == pc2.node_type and pc1.node_type == 'product':
        cw_val = 0
        for i, c1 in enumerate(pc1.children):
            for j, c2 in enumerate(pc2.children):
                if c1.scope != c2.scope:
                    continue
                cw_val += cw_dist(c1, c2, memo=memo)
        
        memo[(pc1, pc2)] = cw_val
        return memo[(pc1, pc2)]

    # If pc1 and pc2 are inputs
    if pc1.node_type == pc2.node_type and pc1.node_type == 'input':
        memo[(pc1, pc2)] = abs(pc1.params[0] - pc2.params[0])
        return memo[(pc1, pc2)]

    # Here, we know that either pc1 or pc2 is a sum node, while the other is either a product or input node.
    if pc1.node_type == 'sum':
        cw_val = 0
        for i, c1 in enumerate(pc1.children):
            cw_val += pc1.params[i] * cw_dist(c1, pc2, memo=memo)
        
        memo[(pc1, pc2)] = cw_val
        return memo[(pc1, pc2)]
        
    if pc2.node_type == 'sum':
        cw_val = 0
        for j, c2 in enumerate(pc2.children):
            cw_val += pc2.params[j] * cw_dist(pc1, c2, memo=memo)
        
        memo[(pc1, pc2)] = cw_val
        return memo[(pc1, pc2)]

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