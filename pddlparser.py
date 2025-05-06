import pddlgym
from pddlgym.structs import Literal
from pddlgym.structs import Predicate
from itertools import product

class Model:
    def __init__(self, predicates, actions, effects, preconditions):
        self.predicates = predicates
        self.actions = actions
        self.effects = effects
        self.preconditions = preconditions

def are_models_compatible(m1: Model, m2: Model):
    if set([p.pddl_str() for p in m1.predicates]) != set([p.pddl_str() for p in m2.predicates]):
        return False
    if set([p.pddl_str() for p in m1.actions]) != set([p.pddl_str() for p in m2.actions]):
        return False

    return True

# Given PPDDL domain and problem filenames, return model
def load_and_ground_pddl(domain_fname: str, problem_dir: str) -> Model:
    env = pddlgym.core.PDDLEnv(domain_file=domain_fname, problem_dir=problem_dir, operators_as_actions=True)
    domain = pddlgym.parser.PDDLDomainParser(domain_fname)
    state, debug = env.reset()

    # Get object types from the domain or problem
    object_types = domain.types  # Mapping of type names to Type objects
    type_hierarchy = domain.type_hierarchy  # For inheritance, if any
    problem_objects = env._problem.objects  # Dictionary of object -> type

    # Function to ground all predicates
    def ground_all_predicates(domain, problem_objects, type_hierarchy):
        grounded_predicates = []
        
        # Access all predicate schemas from the domain
        predicate_schemas = domain.predicates  # Dictionary of name -> Predicate
        
        for pred_name, predicate in predicate_schemas.items():
            # Get the types of the predicate's parameters
            param_types = predicate.var_types  # List of types (e.g., [block, block] for 'on')
            
            # Get valid objects for each parameter
            valid_objects_per_param = [
                get_valid_objects_for_predicate(param_type, problem_objects, type_hierarchy)
                for param_type in param_types
            ]
            
            # Generate all type-valid combinations
            for obj_combination in product(*valid_objects_per_param):
                grounded_predicate = Literal(predicate, obj_combination)
                grounded_predicates.append(grounded_predicate)
        
        return grounded_predicates
    
    # Function to get valid objects for a predicate parameter based on its type
    def get_valid_objects_for_predicate(param_type, problem_objects, type_hierarchy):
        valid_objects = []
        for obj in problem_objects:
            # Check if obj_type matches param_type or is a subtype
            if obj.var_type == param_type or (obj.var_type in type_hierarchy and param_type in type_hierarchy[obj.var_type]):
                valid_objects.append(obj)
        return valid_objects

    # Function to get the grounded effect of an action
    def get_grounded_effect(grounded_action, domain):
        name = grounded_action.predicate.name
        operator = domain.operators[name]
        variables = operator.params
        objects = grounded_action.variables
        
        # Get the effects
        effect = operator.effects  # Could be ProbabilisticEffect or conjunction
        
        # Create substitution from variables to grounded objects
        substitution = dict(zip([v.name for v in variables], objects))
        
        # Ground the effect recursively
        grounded_effect = ground_effect(effect, substitution)
        return grounded_effect
    
    # Function to get the grounded precondition of an action
    def get_grounded_precondition(grounded_action, domain):
        name = grounded_action.predicate.name
        operator = domain.operators[name]
        variables = operator.params
        objects = grounded_action.variables
        
        # Get the effects
        precondition = operator.preconds  # Could be ProbabilisticEffect or conjunction
        
        # Create substitution from variables to grounded objects
        substitution = dict(zip([v.name for v in variables], objects))
        
        # Ground the effect recursively
        grounded_precondition = ground_effect(precondition, substitution)
        return grounded_precondition
    
    # Recursive function to ground an effect
    def ground_effect(effect, substitution):
        # Base case: If it's a Literal, ground it
        if isinstance(effect, pddlgym.structs.Literal):
            grounded_args = tuple(substitution.get(arg.name, arg) for arg in effect.variables)
            return pddlgym.structs.Literal(effect.predicate, grounded_args)
        
        # Probabilistic effect: Recurse over outcomes
        elif type(effect) == pddlgym.structs.ProbabilisticEffect:
            grounded_outcomes = [ground_effect(outcome, substitution) for outcome in effect.literals]
            return pddlgym.structs.ProbabilisticEffect(grounded_outcomes, effect.probabilities)
            #return [(prob, outcome) for prob, outcome in zip(effect.probabilities, grounded_outcomes)]
        
        # Conjunction: Recurse over literals or sub-conjunctions
        elif type(effect) == pddlgym.structs.LiteralConjunction:
            grounded_literals = [ground_effect(lit, substitution) for lit in effect.literals]
            return pddlgym.structs.LiteralConjunction(grounded_literals)
        
        # Fallback: Throw error if unrecognized
        else:
            raise Exception(f"Effect type {type(grounded_effect)} not recognized!")



    # Generate all grounded predicates
    # This represents the state of the world
    grounded_predicates = ground_all_predicates(domain, problem_objects, type_hierarchy)

    # Now that we have all grounded predicates, we want to ground all actions
    grounded_actions = list(env.action_space.all_ground_literals(state))

    # Map each grounded action to its precondition and effect
    grounded_effects = {a: get_grounded_effect(a, domain) for a in grounded_actions}
    grounded_preconditions = {a: get_grounded_precondition(a, domain) for a in grounded_actions}

    return Model(grounded_predicates, grounded_actions, grounded_effects, grounded_preconditions)