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

from pydantic import BaseModel, ConfigDict, Field


# -------------------------
# chat: partner affect meta
# -------------------------

PartnerAffectLabel = Literal["joy", "sadness", "anger", "fear", "neutral"]


class PartnerResponsePolicy(BaseModel):
    """パートナーAIの行動方針ノブ（/api/chat で同期更新する）。"""

    model_config = ConfigDict(extra="forbid")

    cooperation: float = Field(..., ge=0.0, le=1.0)
    refusal_bias: float = Field(..., ge=0.0, le=1.0)
    refusal_allowed: bool


class PartnerAffectMeta(BaseModel):
    """/api/chat の「その瞬間の反応（affect）+ 方針ノブ」を表すメタJSON。"""

    model_config = ConfigDict(extra="forbid")

    reflection_text: str
    partner_affect_label: PartnerAffectLabel
    partner_affect_intensity: float = Field(..., ge=0.0, le=1.0)
    topic_tags: list[str] = Field(default_factory=list, max_length=16)
    salience: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    partner_response_policy: PartnerResponsePolicy


# -------------------------
# Worker: reflection
# -------------------------

class ReflectionOutput(BaseModel):
    """Episode派生のreflection（Workerでも生成する）。"""

    model_config = ConfigDict(extra="forbid")

    reflection_text: str
    partner_affect_label: PartnerAffectLabel
    partner_affect_intensity: float = Field(..., ge=0.0, le=1.0)
    topic_tags: list[str] = Field(default_factory=list, max_length=16)
    salience: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


# -------------------------
# Entities
# -------------------------

class EntityItem(BaseModel):
    """固有名（entity）の1件。"""

    model_config = ConfigDict(extra="forbid")

    type_label: str
    roles: list[str] = Field(default_factory=list, max_length=8)
    name: str
    aliases: list[str] = Field(default_factory=list, max_length=10)
    role: Literal["mentioned"] = "mentioned"
    confidence: float = Field(..., ge=0.0, le=1.0)


class RelationItem(BaseModel):
    """関係（relation）の1件。"""

    model_config = ConfigDict(extra="forbid")

    src: str
    relation: str
    dst: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str


class EntityExtractOutput(BaseModel):
    """entity抽出の出力。"""

    model_config = ConfigDict(extra="forbid")

    entities: list[EntityItem] = Field(default_factory=list, max_length=10)
    relations: list[RelationItem] = Field(default_factory=list, max_length=10)


class EntityNamesOnlyOutput(BaseModel):
    """MemoryPack補助: entity名だけ抽出（names only）。"""

    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(default_factory=list, max_length=10)


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

    from_: Optional[int] = Field(default=None, ge=0, alias="from")
    to: Optional[int] = Field(default=None, ge=0)


class FactItem(BaseModel):
    """Fact 1件。"""

    model_config = ConfigDict(extra="forbid")

    subject: FactEntityRef
    predicate: str
    object_text: Optional[str] = None
    object: Optional[FactEntityRef] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    validity: FactValidity


class FactExtractOutput(BaseModel):
    """fact抽出の出力。"""

    model_config = ConfigDict(extra="forbid")

    facts: list[FactItem] = Field(default_factory=list, max_length=5)


# -------------------------
# Loops
# -------------------------

class LoopItem(BaseModel):
    """Open loop 1件。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["open", "closed"]
    due_at: Optional[int] = Field(default=None, ge=0)
    loop_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class LoopExtractOutput(BaseModel):
    """open loop抽出の出力。"""

    model_config = ConfigDict(extra="forbid")

    loops: list[LoopItem] = Field(default_factory=list, max_length=5)


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
    key_events: list[KeyEventItem] = Field(default_factory=list, max_length=5)
    bond_state: str


class PersonSummaryOutput(BaseModel):
    """人物サマリ。"""

    model_config = ConfigDict(extra="forbid")

    summary_text: str
    favorability_score: float = Field(..., ge=0.0, le=1.0)
    favorability_reasons: list[KeyEventItem] = Field(default_factory=list, max_length=5)
    key_events: list[KeyEventItem] = Field(default_factory=list, max_length=5)
    notes: Optional[str] = None


class TopicSummaryOutput(BaseModel):
    """トピックサマリ。"""

    model_config = ConfigDict(extra="forbid")

    summary_text: str
    key_events: list[KeyEventItem] = Field(default_factory=list, max_length=5)
    notes: Optional[str] = None


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

