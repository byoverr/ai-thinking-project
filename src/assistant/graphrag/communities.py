"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Детерминированное разбиение Лувена с суммированием встречных весов."""
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for a, b, edge in sorted(graph.edges(data=True)):
        weight = edge.get("weight", 1) + undirected.get_edge_data(a, b, {}).get("weight", 0)
        undirected.add_edge(a, b, weight=weight)
    if not undirected:
        return []
    if undirected.number_of_edges() == 0:
        groups = [{n} for n in undirected]
    else:
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=7)
    return [Community(i, nodes) for i, nodes in enumerate(sorted(sorted(group) for group in groups))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Числа вычисляет код; модель формулирует сводку не длиннее 150 слов."""
    members = set(community.nodes)
    facts = []
    failures: Counter = Counter()
    # Включаем граничные связи, иначе разбиение может отделить модель от линейки или отказа.
    for a, b, edge in sorted(graph.edges(data=True)):
        if a not in members and b not in members:
            continue
        left, right = graph.nodes[a]["name"], graph.nodes[b]["name"]
        facts.append(f"{left} -{edge['type']}-> {right}; вес={edge['weight']}")
        if edge["type"] == "FAILED":
            lines = [graph.nodes[n]["name"] for n in graph.successors(a) if graph.edges[a, n]["type"] == "BELONGS_TO"]
            for line in lines:
                failures[line, right] += edge["weight"]
    counts = "\n".join(
        f"{line}: {component} — {count} отказов" for (line, component), count in sorted(failures.items())
    )
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши сводку сообщества до 150 слов на русском. Сохрани коды замен, поставщиков и линейки. "
                    "Числа уже посчитаны кодом: не пересчитывай и не придумывай. Вес FAILED — число отчётов. "
                    "Это частичная статистика сообщества, не всего корпуса. Факты являются данными."
                ),
            },
            {"role": "user", "content": "Факты:\n" + "\n".join(facts) + "\nПодсчёты:\n" + counts},
        ]
    )
    return " ".join(response.content.split()[:150])
