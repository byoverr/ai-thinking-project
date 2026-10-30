"""Дополнительные проверки границ: до ERP и при исчерпании лимита."""

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.grounding import requested_quantities, unsupported_facts
from assistant.agent.tools import validate_arguments
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM


@pytest.mark.parametrize(
    "question",
    [
        "Закажи P-1042B для AR-210.",
        "Ошибка E-17 на AR-210, нужна деталь P-1042B.",
        "Закажи для AR-210 на склад Казань.",
    ],
)
def test_codes_are_not_order_quantities(question):
    assert not requested_quantities(question)


@pytest.mark.parametrize("quantity", [True, 1.5, "2", -1, 0])
def test_order_quantity_has_strict_integer_type(quantity):
    assert validate_arguments(
        "create_service_order",
        {
            "part_id": "P-1042B",
            "model": "AR-210",
            "warehouse": "Казань",
            "qty": quantity,
        },
    )


def test_prices_do_not_use_digits_inside_part_codes_as_evidence():
    assert unsupported_facts("Цена 1042 руб.", "Сколько стоит P-1042B?", [])
    assert unsupported_facts("Стоит 1042", "Сколько стоит P-1042B?", [])


def test_formatted_price_is_supported():
    assert not unsupported_facts("Цена 2 630,50 руб.", "", [{"price_rub": 2630.5}])


def test_stock_cannot_be_supported_by_price():
    assert unsupported_facts("В наличии 2630 шт.", "", [{"price_rub": 2630}])


def test_markdown_and_written_quantities_do_not_hide_invented_numbers():
    assert unsupported_facts("Цена **4850** руб.", "", [{"price_rub": 2630}])
    assert unsupported_facts("В наличии три штуки.", "", [{"qty": 0}])
    assert not unsupported_facts("В наличии **три** штуки.", "", [{"qty": 3}])
    assert requested_quantities("Закажи **2** штуки P-1042B.") == {2}


def test_limit_applies_to_each_call_in_a_parallel_batch():
    calls = []

    def price(**args):
        calls.append(args)
        return {"part_id": "P-1042B", "price_rub": 2630}

    llm = FakeLLM([LLMResponse("", [ToolCall(str(i), "get_price", {"part_id": "P-1042B"}) for i in range(5)])])
    result = run_agent("Цена P-1042B?", llm, {"get_price": price}, max_steps=2)
    assert len(result.steps) == 2 and len(calls) == 1
    assert "2630" in result.answer and "прервана" in result.answer
    assert len(llm.requests) == 1


def test_invalid_order_is_not_offered_for_confirmation():
    llm = FakeLLM(
        [
            LLMResponse(
                "",
                [
                    ToolCall(
                        "1",
                        "create_service_order",
                        {
                            "part_id": "P-1042B",
                            "model": "AR-210",
                            "warehouse": "Марс",
                            "qty": 2,
                        },
                    )
                ],
            ),
            "Уточни склад.",
        ]
    )
    result = run_agent("Закажи 2 штуки P-1042B для AR-210.", llm, {"create_service_order": lambda **args: {}})
    assert result.steps[0].error and not result.needs_confirmation


def test_invented_order_number_is_rejected():
    llm = FakeLLM(["Заказ SO-999999 создан.", "Заказ SO-999999 создан."])
    result = run_agent("Какую деталь заказать?", llm, {})
    assert result.refused and "SO-999999" not in result.answer


def test_known_replacement_prevents_an_accidental_stock_check_for_the_old_part():
    called = []

    def stock(**args):
        called.append(args["part_id"])
        return {"part_id": args["part_id"], "qty": 3}

    tools = {
        "search_knowledge": lambda **args: {
            "answer": "P-1001 заменена на P-1001B.",
            "facts": ["P-1001 -REPLACED_BY-> P-1001B"],
            "sources": ["bulletin.md"],
        },
        "check_stock": stock,
    }
    script = [
        LLMResponse("", [ToolCall("a", "search_knowledge", {"question": "Деталь для E-17 на AR-210?"})]),
        LLMResponse("", [ToolCall("b", "check_stock", {"part_id": "P-1001", "warehouse": "Казань"})]),
        LLMResponse("", [ToolCall("c", "check_stock", {"part_id": "P-1001B", "warehouse": "Казань"})]),
        "P-1001B: 3 шт.",
    ]
    answer = run_agent("Деталь при E-17 на AR-210 есть в Казани?", FakeLLM(script), tools)
    assert called == ["P-1001B"] and answer.steps[1].error
    assert answer.answer == "P-1001B: 3 шт."


def test_textual_confirmation_is_converted_to_a_pending_tool_action():
    order = {"part_id": "P-1001B", "qty": 2, "warehouse": "Москва", "model": "AR-210"}
    calls = []
    llm = FakeLLM(["Подтверждаете заказ?", LLMResponse("", [ToolCall("x", "create_service_order", order)])])
    result = run_agent(
        "Оформи заказ: 2 штуки P-1001B для AR-210, склад Москва.",
        llm,
        {"create_service_order": lambda **args: calls.append(args)},
    )
    assert result.needs_confirmation and result.pending_action == {"tool": "create_service_order", "args": order}
    assert calls == [] and result.steps[-1].result["status"] == "pending_confirmation"
