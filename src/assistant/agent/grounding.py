"""Проверки ответа и количества, независимые от решений языковой модели."""

import json
import re
from decimal import Decimal

from assistant.infra.grounding import find_codes

_NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?(?![\w-])"
MONEY = re.compile(rf"({_NUMBER})\s*(?:руб\w*|₽|р\.)", re.I)
STOCK = re.compile(rf"({_NUMBER})\s*(?:шт\w*|единиц\w*)", re.I)
ORDER_ID = re.compile(r"\bSO-\d+\b", re.I)
WORDS = {
    "один": 1,
    "одну": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}


def _number(value) -> Decimal:
    return Decimal(re.sub(r"\s", "", str(value)).replace(",", "."))


def requested_quantities(question: str) -> set[int]:
    """Не принимаем цифры в кодах моделей/деталей за количество заказа."""
    question = re.sub(r"[*_`]", "", question)
    quantities = {int(n) for n in re.findall(r"(?<![\w.-])(\d+)\s*(?:шт\w*|единиц\w*|детал\w*)", question, re.I)}
    quantities.update(
        int(n) for n in re.findall(r"(?:количеств\w*\s*[:=]?|закажи\w*|заказать)\s+(\d+)(?![\w-])", question, re.I)
    )
    quantities.update(int(n) for n in re.findall(r"(?<![\w.-])(\d+)\s+(?=P-\d{4})", question, re.I))
    for word, value in WORDS.items():
        if re.search(rf"\b{word}\s+(?:шт\w*|единиц\w*|детал\w*)", question, re.I):
            quantities.add(value)
    return quantities


def unsupported_facts(answer: str, question: str, results: list[dict]) -> list[str]:
    """Сверяет коды, суммы, остатки и номера заказов только с доступными наблюдениями."""
    answer = re.sub(r"[*_`]", "", answer)
    question = re.sub(r"[*_`]", "", question)
    for word, value in WORDS.items():
        answer = re.sub(rf"\b{word}\s+(?=(?:шт|единиц))", f"{value} ", answer, flags=re.I)
    evidence = question + "\n" + json.dumps(results, ensure_ascii=False)
    problems = [code for code in find_codes(answer) if code not in set(find_codes(evidence))]
    problems.extend(code for code in ORDER_ID.findall(answer) if code not in ORDER_ID.findall(evidence))
    prices = {_number(r["price_rub"]) for r in results if isinstance(r.get("price_rub"), (int, float))}
    quantities = {_number(r["qty"]) for r in results if type(r.get("qty")) is int}
    prices.update(_number(n) for n in MONEY.findall(question))
    quantities.update(Decimal(n) for n in requested_quantities(question))
    for pattern, allowed, label in ((MONEY, prices, "цена"), (STOCK, quantities, "количество")):
        for n in pattern.findall(answer):
            if _number(n) not in allowed:
                problems.append(f"{label}: {n}")
    # Ловим и «цена: 2000», «остаток — 7», когда модель опускает единицы.
    for label, allowed in ((r"цен\w*|стоимость|стоит", prices), (r"остаток|в наличии|на складе", quantities)):
        for n in re.findall(rf"(?:{label})\s*[:—–=-]?\s*({_NUMBER})", answer, re.I):
            if _number(n) not in allowed:
                problems.append(f"неподтверждённое число: {n}")
    return list(dict.fromkeys(problems))
