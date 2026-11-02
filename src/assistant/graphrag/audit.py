"""Аудит фильтра извлечения по кешу: не выполняет новых запросов к модели."""

import json

from assistant.graphrag.extraction import _Extraction, extract_chunk
from assistant.infra.chunking import load_chunks
from assistant.infra.config import Settings, vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Entity, Relation


class CacheOnlyLLM(LLMClient):
    def __init__(self, settings):
        super().__init__(
            settings.llm_base_url,
            settings.llm_model,
            settings.embed_model,
            cache_dir=settings.artifacts_dir / "llm_cache",
        )
        self.last_content = ""

    def _post(self, path, body):
        raise RuntimeError("Нет ответа в кеше: сначала завершите make index с текущим кодом.")

    def chat(self, *args, **kwargs):
        response = super().chat(*args, **kwargs)
        self.last_content = response.content
        return response


def main():
    settings = Settings.from_env()
    llm = CacheOnlyLLM(settings)
    vocab = vocabulary()
    dropped_entities = dropped_relations = empty_chunks = malformed_chunks = 0
    rows = []
    for chunk in load_chunks(settings.data_dir / "corpus"):
        result = extract_chunk(chunk, llm)
        empty_chunks += not result.entities and not result.relations
        try:
            raw = _Extraction.model_validate_json(llm.last_content)
        except ValueError:
            malformed_chunks += 1
            continue
        codes = set(find_codes(chunk.text))

        def confirmed(kind, name, codes=codes):
            canonical = canonical_name(kind, name, vocab)
            if canonical is None or kind in CODE_TYPES and canonical not in codes:
                return None
            return Entity(kind, canonical)

        bad_entities = sum(confirmed(e.type, e.name) is None for e in raw.entities)
        bad_relations = 0
        rejected = []
        for relation in raw.relations:
            a = confirmed(relation.subject_type, relation.subject)
            b = confirmed(relation.object_type, relation.object)
            reason = None
            if a is None or b is None:
                reason = "Конец связи не подтверждён текстом или словарём"
            elif not Relation(a, relation.type, b).matches_schema():
                reason = "Типы или направление связи нарушают схему"
            if reason:
                bad_relations += 1
                rejected.append({**relation.model_dump(), "reason": reason})
        dropped_entities += bad_entities
        dropped_relations += bad_relations
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "entities": len(result.entities),
                "relations": len(result.relations),
                "dropped_entities": bad_entities,
                "dropped_relations": bad_relations,
                "rejected_relations": rejected,
            }
        )
    report = {
        "dropped_entities": dropped_entities,
        "dropped_relations": dropped_relations,
        "empty_chunks": empty_chunks,
        "malformed_chunks": malformed_chunks,
        "rows": rows,
    }
    path = settings.artifacts_dir / "extraction_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
