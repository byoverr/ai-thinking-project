"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет источники, считает разные чанки и разрешает встречные замены."""
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        relations = [r for r in result.relations if r.matches_schema()]
        entities = [*result.entities, *(e for r in relations for e in (r.subject, r.object))]
        for entity in entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
            sources = graph.nodes[entity.node_id]["sources"]
            if result.chunk_id not in sources:
                sources.append(result.chunk_id)
        for relation in relations:
            a, b = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, type=relation.type, weight=0, sources=[])
            edge = graph.edges[a, b]
            edge["sources"] = sorted(set(edge["sources"]) | {result.chunk_id})
            edge["weight"] = len(edge["sources"])
    for a, b in list(graph.edges):
        if a >= b or not graph.has_edge(a, b) or not graph.has_edge(b, a):
            continue
        forward, reverse = graph.edges[a, b], graph.edges[b, a]
        if forward["type"] != "REPLACED_BY" or reverse["type"] != "REPLACED_BY":
            continue
        graph.graph["conflicts"] += 1
        wa, wb = forward["weight"], reverse["weight"]
        if wa <= wb:
            graph.remove_edge(a, b)
        if wb <= wa:
            graph.remove_edge(b, a)
    for _, attrs in graph.nodes(data=True):
        attrs["sources"].sort()
    return graph
