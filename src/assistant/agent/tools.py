"""Пять инструментов агента и проверка аргументов. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART = {"type": "string", "pattern": r"^P-[0-9]{4}[A-Z]?$"}
WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
QUESTION = {"question": {"type": "string", "minLength": 1}}


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Поиск узла, детали и всей цепочки замен по модели и ошибке. "
        "Сначала используй его, если номер детали неизвестен.",
        QUESTION,
    ),
    _schema("summarize_topic", "Обзорные вопросы: частые отказы, поставщики и линейки оборудования.", QUESTION),
    _schema(
        "check_stock",
        "Остаток конкретной детали на складе и признак снятия с производства.",
        {"part_id": PART, "warehouse": WAREHOUSE},
    ),
    _schema("get_price", "Цена конкретной детали в рублях и признак снятия с производства.", {"part_id": PART}),
    _schema(
        "create_service_order",
        "Подготовить заказ только по явному запросу пользователя. "
        "Количество пользователь должен назвать сам; создание требует подтверждения.",
        {
            "part_id": PART,
            "warehouse": WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "pattern": r"^(AR|BZ|GS)-[0-9]{3}$"},
        },
    ),
]


def validate_arguments(name: str, args: dict) -> str | None:
    """Проверка выполняется и агентом (до подтверждения), и самими инструментами."""
    schema = next((s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name), None)
    if schema is None:
        return f"Неизвестный инструмент: {name}"
    if not isinstance(args, dict) or set(args) != set(schema["required"]):
        return "Нечитаемые аргументы или неверный набор полей. Используй схему инструмента."
    for key, spec in schema["properties"].items():
        value = args[key]
        if spec["type"] == "string":
            if not isinstance(value, str) or not value.strip():
                return f"{key}: требуется непустая строка"
            if "pattern" in spec and re.fullmatch(spec["pattern"], value) is None:
                return f"{key}: неверный формат кода"
            if "enum" in spec and value not in spec["enum"]:
                return f"{key}: выбери склад из {', '.join(WAREHOUSES)}"
        elif type(value) is not int or value < spec.get("minimum", 0):
            return f"{key}: требуется положительное целое число"
    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Аргументы проверяются до запросов; временные ошибки передаются агенту."""

    def search(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    functions = {
        "search_knowledge": search,
        "summarize_topic": summarize,
        "check_stock": erp.get_stock,
        "get_price": erp.get_price,
        "create_service_order": erp.create_order,
    }

    def wrap(name: str, function: Callable) -> Callable[..., dict]:
        def invoke(**kwargs) -> dict:
            error = validate_arguments(name, kwargs)
            if error:
                return {"error": error}
            try:
                return function(**kwargs)
            except ErpNotFound as exc:
                return {"error": f"Данных в ERP нет: {exc}", "error_type": "not_found"}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}", "error_type": "rejected"}

        return invoke

    return {name: wrap(name, function) for name, function in functions.items()}
