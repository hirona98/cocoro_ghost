"""
LLM Structured Outputs 用のスキーマ定義

このモジュールは「LLMが返すJSON」をPydanticモデルとして一元管理する。

方針:
- "JSONとして扱う出力" は全てここに集約する（プロンプト文よりスキーマを正にする）
- extra=forbid で未知キーを禁止し、出力のブレを減らす
- 数値は可能な限り 0..1 / epoch seconds 等、内部で扱いやすい型へ寄せる
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _require_0_1(value: float, field_name: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def _require_nonneg_int(value: Optional[int], field_name: str) -> Optional[int]:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _require_max_len(values: list, max_len: int, field_name: str) -> list:
    if values is None:
        raise ValueError(f"{field_name} must be a list")
    if len(values) > max_len:
        raise ValueError(f"{field_name} must have at most {max_len} items")
    return values


# -------------------------
# chat: partner affect meta
# -------------------------

PartnerAffectLabel = Literal["joy", "sadness", "anger", "fear", "neutral"]


class PartnerResponsePolicy(BaseModel):
    """パートナーAIの行動方針ノブ（/api/chat で同期更新する）。"""

    model_config = ConfigDict(extra="forbid")

    cooperation: float = Field(...)
    refusal_bias: float = Field(...)
    refusal_allowed: bool

    @field_validator("cooperation", "refusal_bias")
    @classmethod
    def _validate_range(cls, v, info):
        return _require_0_1(float(v), info.field_name)


class PartnerAffectMeta(BaseModel):
    """/api/chat の「その瞬間の反応（affect）+ 方針ノブ」を表すメタJSON。"""

    model_config = ConfigDict(extra="forbid")

    reflection_text: str
    partner_affect_label: PartnerAffectLabel
    partner_affect_intensity: float = Field(...)
    topic_tags: list[str] = Field(default_factory=list)
    salience: float = Field(...)
    confidence: float = Field(...)
    partner_response_policy: PartnerResponsePolicy

    @field_validator("partner_affect_intensity", "salience", "confidence")
    @classmethod
    def _validate_range(cls, v, info):
        return _require_0_1(float(v), info.field_name)

    @field_validator("topic_tags")
    @classmethod
    def _validate_topic_tags_len(cls, v):
        return _require_max_len(v, 16, "topic_tags")


# -------------------------
# Worker: reflection
# -------------------------

class ReflectionOutput(BaseModel):
    """Episode派生のreflection（Workerでも生成する）。"""

    model_config = ConfigDict(extra="forbid")

    reflection_text: str
    partner_affect_label: PartnerAffectLabel
    partner_affect_intensity: float = Field(...)
    topic_tags: list[str] = Field(default_factory=list)
    salience: float = Field(...)
    confidence: float = Field(...)

    @field_validator("partner_affect_intensity", "salience", "confidence")
    @classmethod
    def _validate_range(cls, v, info):
        return _require_0_1(float(v), info.field_name)

    @field_validator("topic_tags")
    @classmethod
    def _validate_topic_tags_len(cls, v):
        return _require_max_len(v, 16, "topic_tags")


# -------------------------
# Entities
# -------------------------

class EntityItem(BaseModel):
    """固有名（entity）の1件。"""

    model_config = ConfigDict(extra="forbid")

    type_label: str
    roles: list[str] = Field(default_factory=list)
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: Literal["mentioned"] = "mentioned"
    confidence: float = Field(...)

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v):
        return _require_0_1(float(v), "confidence")

    @field_validator("roles")
    @classmethod
    def _validate_roles_len(cls, v):
        return _require_max_len(v, 8, "roles")

    @field_validator("aliases")
    @classmethod
    def _validate_aliases_len(cls, v):
        return _require_max_len(v, 10, "aliases")


class RelationItem(BaseModel):
    """関係（relation）の1件。"""

    model_config = ConfigDict(extra="forbid")

    src: str
    relation: str
    dst: str
    confidence: float = Field(...)
    evidence: str

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v):
        return _require_0_1(float(v), "confidence")


class EntityExtractOutput(BaseModel):
    """entity抽出の出力。"""

    model_config = ConfigDict(extra="forbid")

    entities: list[EntityItem] = Field(default_factory=list)
    relations: list[RelationItem] = Field(default_factory=list)

    @field_validator("entities")
    @classmethod
    def _validate_entities_len(cls, v):
        return _require_max_len(v, 10, "entities")

    @field_validator("relations")
    @classmethod
    def _validate_relations_len(cls, v):
        return _require_max_len(v, 10, "relations")


class EntityNamesOnlyOutput(BaseModel):
    """MemoryPack補助: entity名だけ抽出（names only）。"""

    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(default_factory=list)

    @field_validator("names")
    @classmethod
    def _validate_names_len(cls, v):
        return _require_max_len(v, 10, "names")


# -------------------------
# Facts
# -------------------------

class FactEntityRef(BaseModel):
    """Factの主語/目的語としてのEntity参照。"""

    model_config = ConfigDict(extra="forbid")

    type_label: str
    name: str


class FactValidity(BaseModel):
    """Factの有効期間（epoch seconds or null）。"""

    # NOTE: "from" は予約語のため from_ を使う（JSONキーは alias で "from" に寄せる）。
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: Optional[int] = Field(default=None, alias="from")
    to: Optional[int] = Field(default=None)

    @field_validator("from_", "to")
    @classmethod
    def _validate_nonneg(cls, v, info):
        return _require_nonneg_int(v, info.field_name)


class FactItem(BaseModel):
    """Fact 1件。"""

    model_config = ConfigDict(extra="forbid")

    subject: FactEntityRef
    predicate: str
    object_text: Optional[str] = None
    object: Optional[FactEntityRef] = None
    confidence: float = Field(...)
    validity: FactValidity

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v):
        return _require_0_1(float(v), "confidence")


class FactExtractOutput(BaseModel):
    """fact抽出の出力。"""

    model_config = ConfigDict(extra="forbid")

    facts: list[FactItem] = Field(default_factory=list)

    @field_validator("facts")
    @classmethod
    def _validate_facts_len(cls, v):
        return _require_max_len(v, 5, "facts")


# -------------------------
# Loops
# -------------------------

class LoopItem(BaseModel):
    """Open loop 1件。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["open", "closed"]
    due_at: Optional[int] = Field(default=None)
    loop_text: str
    confidence: float = Field(...)

    @field_validator("due_at")
    @classmethod
    def _validate_due_at(cls, v):
        return _require_nonneg_int(v, "due_at")

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v):
        return _require_0_1(float(v), "confidence")


class LoopExtractOutput(BaseModel):
    """open loop抽出の出力。"""

    model_config = ConfigDict(extra="forbid")

    loops: list[LoopItem] = Field(default_factory=list)

    @field_validator("loops")
    @classmethod
    def _validate_loops_len(cls, v):
        return _require_max_len(v, 5, "loops")


# -------------------------
# Summaries
# -------------------------

class KeyEventItem(BaseModel):
    """サマリの根拠（unit_id + 短い理由）。"""

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    why: str


class BondSummaryOutput(BaseModel):
    """絆サマリ（BondSummary）。"""

    model_config = ConfigDict(extra="forbid")

    summary_text: str
    key_events: list[KeyEventItem] = Field(default_factory=list)
    bond_state: str

    @field_validator("key_events")
    @classmethod
    def _validate_key_events_len(cls, v):
        return _require_max_len(v, 5, "key_events")


class PersonSummaryOutput(BaseModel):
    """人物サマリ。"""

    model_config = ConfigDict(extra="forbid")

    summary_text: str
    favorability_score: float = Field(...)
    favorability_reasons: list[KeyEventItem] = Field(default_factory=list)
    key_events: list[KeyEventItem] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("favorability_score")
    @classmethod
    def _validate_favorability_score(cls, v):
        return _require_0_1(float(v), "favorability_score")

    @field_validator("favorability_reasons")
    @classmethod
    def _validate_favorability_reasons_len(cls, v):
        return _require_max_len(v, 5, "favorability_reasons")

    @field_validator("key_events")
    @classmethod
    def _validate_key_events_len(cls, v):
        return _require_max_len(v, 5, "key_events")


class TopicSummaryOutput(BaseModel):
    """トピックサマリ。"""

    model_config = ConfigDict(extra="forbid")

    summary_text: str
    key_events: list[KeyEventItem] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("key_events")
    @classmethod
    def _validate_key_events_len(cls, v):
        return _require_max_len(v, 5, "key_events")


def partner_affect_meta_tool() -> dict:
    """/api/chat 用: メタJSONを回収するための function tool 定義を返す。

    NOTE:
    - ここは Chat Completions の tool calling を使う。
    - response_format(json_schema) だと本文ストリームと両立しないため、メタだけをツール経由で厳格化する。
    """

    # PydanticのJSON Schemaを tool parameters として使う。
    # OpenAIの tool schema は JSON Schema なので、この形式で十分。
    schema = PartnerAffectMeta.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": "cocoro_emit_partner_affect_meta",
            "description": "会話の内部メタ（affect/重要度/方針）を1回だけ報告する。",
            "parameters": schema,
        },
    }
