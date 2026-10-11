"""Discrete Bayesian network: structure learning, CPDs, queries and alert context."""

from metroguard.models.pgmpy_import_fallback import install_statsmodels_api_fallback

install_statsmodels_api_fallback()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pgmpy.causal_discovery import HillClimbSearch  # noqa: E402
from pgmpy.inference import VariableElimination  # noqa: E402
from pgmpy.models import DiscreteBayesianNetwork  # noqa: E402
from pgmpy.parameter_estimator import DiscreteBayesianEstimator  # noqa: E402


def categorical_node_table(
    states: pd.DataFrame, nodes: list[str], state_counts: dict[str, int]
) -> pd.DataFrame:
    """Selected node columns as categoricals that list every possible state."""
    return pd.DataFrame(
        {
            node: pd.Categorical(
                states[node].to_numpy(), categories=list(range(state_counts[node]))
            )
            for node in nodes
        },
        index=states.index,
    )


def learn_structure(
    data: pd.DataFrame, scoring_method: str, max_indegree: int
) -> list[tuple[str, str]]:
    """Hill-climb search with a BIC score; returns the learned directed edges."""
    search = HillClimbSearch(
        scoring_method=scoring_method,
        max_indegree=max_indegree,
        return_type="dag",
        show_progress=False,
    )
    return sorted(search.fit(data).causal_graph_.edges())


def fit_cpds(
    edges: list[tuple[str, str]], data: pd.DataFrame, equivalent_sample_size: float
) -> DiscreteBayesianNetwork:
    """Conditional probability tables by Bayesian (BDeu) estimation on fit windows."""
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(data.columns)
    state_names = {c: list(data[c].cat.categories) for c in data.columns}
    estimator = DiscreteBayesianEstimator(
        state_names=state_names, prior_type="BDeu", equivalent_sample_size=equivalent_sample_size
    )
    model.fit(data, estimator=estimator)
    return model


def conditional_table(
    model: DiscreteBayesianNetwork, target: str, evidence_variable: str
) -> pd.DataFrame:
    """P(target | evidence_variable = s) for every state s of the evidence variable."""
    inference = VariableElimination(model)
    rows = {}
    for state in model.get_cpds(evidence_variable).state_names[evidence_variable]:
        factor = inference.query([target], evidence={evidence_variable: state}, show_progress=False)
        rows[f"{evidence_variable}={state}"] = factor.values
    states = model.get_cpds(target).state_names[target]
    return pd.DataFrame(rows, index=[f"{target}={s}" for s in states])


def strongest_edge(
    model: DiscreteBayesianNetwork, data: pd.DataFrame
) -> tuple[tuple[str, str], float]:
    """The learned edge whose endpoints have the highest empirical mutual information (nats)."""

    def mutual_information(parent: str, child: str) -> float:
        joint = pd.crosstab(data[parent], data[child]).to_numpy(dtype=float)
        joint /= joint.sum()
        outer = np.outer(joint.sum(axis=1), joint.sum(axis=0))
        mask = joint > 0
        return float((joint[mask] * np.log(joint[mask] / outer[mask])).sum())

    scored = {edge: mutual_information(*edge) for edge in model.edges()}
    best = max(scored, key=scored.get)
    return best, scored[best]


def d_connected(model: DiscreteBayesianNetwork, start: str, end: str, observed: list[str]) -> bool:
    """True when `end` is reachable from `start` by an active trail given `observed`."""
    return end in model.active_trail_nodes(start, observed=observed)[start]


def local_independence_cases(model: DiscreteBayesianNetwork) -> list[dict]:
    """Node vs its parents, vs a non-parent non-descendant, and vs a descendant."""
    for node in model.nodes():
        parents = sorted(model.get_parents(node))
        descendants = sorted(model.get_children(node))
        non_descendants = set(model.nodes()) - {node} - _all_descendants(model, node)
        others = sorted(non_descendants - set(parents))
        if parents and descendants and others:
            other = others[0]
            return [
                {
                    "case": "node and a parent",
                    "pair": (node, parents[0]),
                    "given": [],
                    "d_connected": d_connected(model, node, parents[0], []),
                },
                {
                    "case": "node and a non-parent non-descendant, given its parents",
                    "pair": (node, other),
                    "given": parents,
                    "d_connected": d_connected(model, node, other, parents),
                },
                {
                    "case": "node and a descendant",
                    "pair": (node, descendants[0]),
                    "given": [],
                    "d_connected": d_connected(model, node, descendants[0], []),
                },
            ]
    return []


def _all_descendants(model: DiscreteBayesianNetwork, node: str) -> set[str]:
    found: set[str] = set()
    stack = list(model.get_children(node))
    while stack:
        current = stack.pop()
        if current not in found:
            found.add(current)
            stack.extend(model.get_children(current))
    return found


def structure_patterns(model: DiscreteBayesianNetwork) -> list[dict]:
    """A chain, fork and collider, preferring those where evidence changes d-connection."""
    edges = set(model.edges())
    patterns: list[dict] = []
    for kind in ("chain", "fork", "collider"):
        candidates: list[dict] = []
        for middle in sorted(model.nodes()):
            parents, children = (
                sorted(model.get_parents(middle)),
                sorted(model.get_children(middle)),
            )
            pairs = {
                "chain": [(p, c) for p in parents for c in children],
                "fork": [(a, b) for i, a in enumerate(children) for b in children[i + 1 :]],
                "collider": [(a, b) for i, a in enumerate(parents) for b in parents[i + 1 :]],
            }[kind]
            pairs = [(a, b) for a, b in pairs if (a, b) not in edges and (b, a) not in edges]
            for a, b in pairs:
                candidates.append(
                    {
                        "kind": kind,
                        "ends": (a, b),
                        "middle": middle,
                        "connected_without_evidence": d_connected(model, a, b, []),
                        "connected_given_middle": d_connected(model, a, b, [middle]),
                    }
                )
        informative = [
            c for c in candidates if c["connected_without_evidence"] != c["connected_given_middle"]
        ]
        patterns.extend((informative or candidates)[:1])
    return patterns


def markov_blanket_probability(
    model: DiscreteBayesianNetwork, states: pd.DataFrame, variable: str
) -> np.ndarray:
    """P(variable = observed state | its Markov blanket), per row, from the CPD factors."""
    cardinality = model.get_cardinality(variable)
    factors = [model.get_cpds(n) for n in [variable, *model.get_children(variable)]]
    codes = {c: states[c].to_numpy(dtype=int) for c in states.columns}
    joint = np.ones((len(states), cardinality))
    for state in range(cardinality):
        for cpd in factors:
            index = tuple(
                np.full(len(states), state) if v == variable else codes[v] for v in cpd.variables
            )
            joint[:, state] *= cpd.values[index]
    observed = joint[np.arange(len(states)), codes[variable]]
    return observed / joint.sum(axis=1)


def blanket_surprisal(
    model: DiscreteBayesianNetwork, states: pd.DataFrame, variable: str
) -> np.ndarray:
    """-log P(observed state | Markov blanket) per row, in nats."""
    return -np.log(markov_blanket_probability(model, states, variable))
