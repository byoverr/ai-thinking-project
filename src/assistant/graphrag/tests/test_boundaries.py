"""Пустые графы, циклические замены и невалидные частичные ответы."""

import networkx as nx

from assistant.graphrag.communities import detect_communities
from assistant.graphrag.search import global_search, local_search
from assistant.infra.schemas import Community, KnowledgeBase
from assistant.infra.testing import FakeLLM


def test_empty_and_edgeless_graphs_have_valid_partitions():
    assert detect_communities(nx.DiGraph()) == []
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c"])
    assert [c.nodes for c in detect_communities(graph)] == [["a"], ["b"], ["c"]]


def test_replacement_cycle_terminates():
    graph = nx.DiGraph()
    for part in ["P-1001", "P-1002", "P-1003"]:
        graph.add_node(f"Part:{part}", type="Part", name=part, sources=["a.md#0"])
    for a, b in [("P-1001", "P-1002"), ("P-1002", "P-1003"), ("P-1003", "P-1001")]:
        graph.add_edge(f"Part:{a}", f"Part:{b}", type="REPLACED_BY", weight=1, sources=["a.md#0"])
    result = local_search("Чем заменена P-1001?", KnowledgeBase(graph, [], {}), FakeLLM(["Замены противоречивы."]))
    assert len(result.facts) == 3 and result.sources == ["a.md"]


def test_map_rejects_non_objects_and_scores_outside_schema():
    for raw in ["[]", "null", '{"answer":"да","score":11}', '{"answer":"да","score":true}']:
        kb = KnowledgeBase(nx.DiGraph(), [Community(0, [], "сводка")], {})
        assert global_search("Обзор?", kb, FakeLLM([raw])).refused


def test_missing_stock_does_not_turn_a_supported_part_answer_into_a_refusal():
    graph = nx.DiGraph()
    graph.add_node("Part:P-1001", type="Part", name="P-1001", sources=["a.md#0"])
    graph.add_node("Part:P-1001B", type="Part", name="P-1001B", sources=["a.md#0"])
    graph.add_edge("Part:P-1001", "Part:P-1001B", type="REPLACED_BY", weight=1, sources=["a.md#0"])
    llm = FakeLLM(["Закажите P-1001B. Остаток: в базе знаний не найдено."])
    answer = local_search("Чем заменить P-1001 и есть ли она на складе?", KnowledgeBase(graph, [], {}), llm)
    assert not answer.refused and answer.sources == ["a.md"]


def test_supplier_of_replacement_is_included_for_model_component_question():
    graph = nx.DiGraph()
    for node in ["Model:AR-210", "Part:P-1001", "Part:P-1001B", "Component:пусковое реле", "Supplier:ХолодКомплект"]:
        kind, name = node.split(":", 1)
        graph.add_node(node, type=kind, name=name, sources=["a.md#0"])
    for a, kind, b in [
        ("Model:AR-210", "HAS_PART", "Part:P-1001"),
        ("Part:P-1001", "IS_A", "Component:пусковое реле"),
        ("Part:P-1001", "REPLACED_BY", "Part:P-1001B"),
        ("Part:P-1001B", "SUPPLIED_BY", "Supplier:ХолодКомплект"),
    ]:
        graph.add_edge(a, b, type=kind, weight=1, sources=["a.md#0"])
    kb = KnowledgeBase(graph, [], {"Component": ["пусковое реле"]})
    llm = FakeLLM(["ХолодКомплект."])
    answer = local_search("Кто поставляет пусковое реле для AR-210?", kb, llm)
    assert any("SUPPLIED_BY" in fact and "ХолодКомплект" in fact for fact in answer.facts)


def test_model_cannot_drop_the_terminal_replacement_from_local_answer():
    graph = nx.DiGraph()
    for code in ["P-1001", "P-1001B", "P-1001C"]:
        graph.add_node("Part:" + code, type="Part", name=code, sources=["a.md#0"])
    for a, b in [("P-1001", "P-1001B"), ("P-1001B", "P-1001C")]:
        graph.add_edge("Part:" + a, "Part:" + b, type="REPLACED_BY", weight=1, sources=["a.md#0"])
    answer = local_search("Что заказать вместо P-1001?", KnowledgeBase(graph, [], {}), FakeLLM(["Закажите P-1001B."]))
    assert "P-1001C" in answer.text and not answer.refused
    assert "Закажите P-1001B" not in answer.text


def test_invented_code_in_local_generation_does_not_enter_tool_evidence():
    graph = nx.DiGraph()
    graph.add_node("ErrorCode:E-17", type="ErrorCode", name="E-17", sources=["a.md#0"])
    graph.add_node("Procedure:SP-104", type="Procedure", name="SP-104", sources=["a.md#0"])
    graph.add_edge("ErrorCode:E-17", "Procedure:SP-104", type="FIXED_BY", weight=1, sources=["a.md#0"])
    result = local_search("Как устранить E-17?", KnowledgeBase(graph, [], {}), FakeLLM(["По регламенту SP-999."]))
    assert "SP-999" not in result.text and "SP-104" in result.text
