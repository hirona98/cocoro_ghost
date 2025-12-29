"""cocoro_ghost.llm_client

LiteLLM ラッパー。

LLM API 呼び出しを抽象化するクライアントクラス。
現状は `litellm.completion()`（OpenAI の chat.completions 互換）を中心に利用する。
会話生成、JSON 生成、埋め込みベクトル生成、画像認識をサポートする。

設計方針:
- JSONとして扱う出力は Structured Outputs（json_schema / strict）で受け取り、後段で修復しない
- /api/chat の本文はストリームしつつ、同期メタは tool call（function calling）で回収する
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, TypeVar

import litellm
from pydantic import BaseModel

from cocoro_ghost.llm_debug import log_llm_payload


def _truncate_for_log(text: str, limit: int) -> str:
    """ログ出力用にテキストを切り詰める。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _response_to_dict(resp: Any) -> Dict[str, Any]:
    """
    LiteLLM Responseをシリアライズ可能なdictに変換する。
    デバッグやログ出力に使用する。
    """
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "dict"):
        return resp.dict()
    if hasattr(resp, "json"):
        try:
            return json.loads(resp.json())
        except Exception:  # noqa: BLE001
            pass
    return dict(resp) if isinstance(resp, dict) else {"raw": str(resp)}


def _first_choice_content(resp: Any) -> str:
    """
    choices[0].message.contentを取り出すユーティリティ。
    LLMレスポンスから本文テキストを抽出する。
    """
    try:
        choice = resp.choices[0]
        message = getattr(choice, "message", None) or choice["message"]
        content = getattr(message, "content", None) or message["content"]
        # OpenAI形式で content が list の場合もあるため統一
        if isinstance(content, list):
            return "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content]) or ""
        return content or ""
    except Exception:  # noqa: BLE001
        return ""


def _delta_content(resp: Any) -> str:
    """
    ストリーミングチャンクからdelta.contentを取り出す。
    ストリーミング応答の逐次処理に使用する。
    """
    try:
        choice = resp.choices[0]
        delta = getattr(choice, "delta", None) or choice.get("delta")
        if not delta:
            return ""
        content = getattr(delta, "content", None) or delta.get("content")
        if content is None:
            return ""
        # OpenAI形式で content が list の場合もあるため統一
        if isinstance(content, list):
            return "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
        return str(content)
    except Exception:  # noqa: BLE001
        return ""


def _stream_delta_tool_calls(resp: Any) -> list[dict]:
    """ストリーミングチャンクから tool_calls を取り出す。

    OpenAI互換の delta.tool_calls（function calling）の断片が入ってくる。
    ここでは「断片をそのまま返す」だけにし、結合は呼び出し側で行う。
    """
    try:
        choice = resp.choices[0]
        delta = getattr(choice, "delta", None) or choice.get("delta")
        if not delta:
            return []
        tool_calls = getattr(delta, "tool_calls", None) or delta.get("tool_calls")
        if not tool_calls:
            return []
        return list(tool_calls) if isinstance(tool_calls, list) else []
    except Exception:  # noqa: BLE001
        return []


T_Model = TypeVar("T_Model", bound=BaseModel)


class LlmClient:
    """
    LLM APIクライアント。
    LiteLLMを使用してLLM APIを呼び出し、会話応答やJSON生成を行う。
    """

    # ログ出力時のプレビュー文字数
    _INFO_PREVIEW_CHARS = 300
    _DEBUG_PREVIEW_CHARS = 5000

    def __init__(
        self,
        model: str,
        embedding_model: str,
        image_model: str,
        api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        image_llm_base_url: Optional[str] = None,
        image_model_api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: int = 4096,
        max_tokens_vision: int = 4096,
        image_timeout_seconds: int = 60,
    ):
        """
        LLMクライアントを初期化する。

        Args:
            model: メインLLMモデル名
            embedding_model: 埋め込みモデル名
            image_model: 画像認識用モデル名
            api_key: LLM APIキー
            embedding_api_key: 埋め込みAPIキー（未指定時はapi_keyを使用）
            llm_base_url: LLM APIベースURL
            embedding_base_url: 埋め込みAPIベースURL
            image_llm_base_url: 画像モデルAPIベースURL
            image_model_api_key: 画像モデルAPIキー
            reasoning_effort: 推論詳細度設定（o1系用）
            max_tokens: 通常時の最大トークン数
            max_tokens_vision: 画像認識時の最大トークン数
            image_timeout_seconds: 画像処理タイムアウト秒数
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.embedding_model = embedding_model
        self.image_model = image_model
        self.api_key = api_key
        self.embedding_api_key = embedding_api_key or api_key
        self.llm_base_url = llm_base_url
        self.embedding_base_url = embedding_base_url
        self.image_llm_base_url = image_llm_base_url
        self.image_model_api_key = image_model_api_key or api_key
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_tokens_vision = max_tokens_vision
        self.image_timeout_seconds = image_timeout_seconds

    def _log_llm_send(
        self,
        *,
        kind: str,
        payload: Any,
        model: str,
        stream: bool,
    ) -> None:
        """LLM送信ログを出す。"""
        # INFOは送受信の到達点だけを出す
        self.logger.info("LLMに送信", extra={"model": model, "stream": stream, "kind": kind})
        if self.logger.isEnabledFor(logging.DEBUG):
            log_llm_payload(self.logger, f"LLM request ({kind})", payload)

    def _log_llm_recv(
        self,
        *,
        kind: str,
        payload: Any,
        model: str,
        stream: bool,
    ) -> None:
        """LLM受信ログを出す。"""
        self.logger.info("LLMから受信", extra={"model": model, "stream": stream, "kind": kind})
        if self.logger.isEnabledFor(logging.DEBUG):
            log_llm_payload(self.logger, f"LLM response ({kind})", payload)

    def _build_completion_kwargs(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[object] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[object] = None,
        parallel_tool_calls: Optional[bool] = None,
        extra_body: Optional[dict] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
    ) -> Dict:
        """
        completion API呼び出し用のkwargsを構築する。
        モデル固有の制約（gpt-5系のtemperature制限等）を考慮する。
        """
        # gpt-5 以降は temperature の制約が厳しい（LiteLLM側で弾かれる）
        # - gpt-5 / gpt-5-mini / gpt-6 ... 等: temperature は 1 のみ
        model_l = (model or "").lower()
        temp = float(temperature)
        m = re.search(r"\bgpt-(\d+)(?:\.(\d+))?\b", model_l)
        if m:
            major = int(m.group(1))
            if major >= 5:
                temp = 1.0

        # 基本パラメータ
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "stream": stream,
        }

        # オプションパラメータを追加
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["api_base"] = base_url
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = bool(parallel_tool_calls)
        if timeout:
            kwargs["timeout"] = timeout

        # extra_body（OpenAI系の追加パラメータ）
        # - reasoning_effort 等はここへ載せる（既存の extra_body があればマージする）
        extra_body_obj: dict = {}
        if self.reasoning_effort:
            extra_body_obj["reasoning_effort"] = self.reasoning_effort
        if extra_body:
            extra_body_obj.update(dict(extra_body))
        if extra_body_obj:
            kwargs["extra_body"] = extra_body_obj

        return kwargs

    def generate_reply_response(
        self,
        system_prompt: str,
        conversation: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[object] = None,
        parallel_tool_calls: Optional[bool] = None,
    ):
        """
        会話応答を生成する（Responseオブジェクト or ストリーム）。
        システムプロンプトと会話履歴からLLMに応答を生成させる。
        """
        # メッセージ配列を構築
        messages = [{"role": "system", "content": system_prompt}]
        for m in conversation:
            messages.append({"role": m["role"], "content": m["content"]})

        kwargs = self._build_completion_kwargs(
            model=self.model,
            messages=messages,
            temperature=temperature,
            api_key=self.api_key,
            base_url=self.llm_base_url,
            max_tokens=self.max_tokens,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        self._log_llm_send(kind="chat", payload=kwargs, model=self.model, stream=stream)
        resp = litellm.completion(**kwargs)
        if not stream:
            self._log_llm_recv(kind="chat", payload=self.response_to_dict(resp), model=self.model, stream=stream)
        return resp

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        response_model: Type[T_Model],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> T_Model:
        """Structured Outputs（json_schema/strict）でJSONを生成し、Pydanticモデルにして返す。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        requested_max_tokens = max_tokens or self.max_tokens
        kwargs = self._build_completion_kwargs(
            model=self.model,
            messages=messages,
            temperature=temperature,
            api_key=self.api_key,
            base_url=self.llm_base_url,
            max_tokens=requested_max_tokens,
            # LiteLLMは pydantic.BaseModel を response_format として受け取れる（json_schema(strict)へ変換される）。
            response_format=response_model,
        )
        self._log_llm_send(kind="structured", payload=kwargs, model=self.model, stream=False)
        resp = litellm.completion(**kwargs)
        self._log_llm_recv(kind="structured", payload=self.response_to_dict(resp), model=self.model, stream=False)
        content = self.response_content(resp)
        try:
            return response_model.model_validate_json(content)
        except Exception as exc:  # noqa: BLE001
            # Structured Outputsのはずだが、失敗時はログに残して即時に落とす（修復しない）。
            self.logger.error(
                "LLM structured output parse failed",
                extra={"model": self.model, "schema": getattr(response_model, "__name__", str(response_model))},
                exc_info=exc,
            )
            self.logger.debug("LLM structured output content: %s", _truncate_for_log(content, self._DEBUG_PREVIEW_CHARS))
            raise

    def generate_embedding(
        self,
        texts: List[str],
        images: Optional[List[bytes]] = None,
    ) -> List[List[float]]:
        """
        テキストの埋め込みベクトルを生成する。
        複数テキストを一括処理し、各テキストに対応するベクトルを返す。
        """
        kwargs = {
            "model": self.embedding_model,
            "input": texts,
        }
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key
        if self.embedding_base_url:
            kwargs["api_base"] = self.embedding_base_url

        self._log_llm_send(kind="embedding", payload=kwargs, model=self.embedding_model, stream=False)
        resp = litellm.embedding(**kwargs)
        self._log_llm_recv(kind="embedding", payload=self.response_to_dict(resp), model=self.embedding_model, stream=False)

        # レスポンス形式に応じてベクトルを抽出
        try:
            return [item["embedding"] for item in resp["data"]]
        except Exception:  # noqa: BLE001
            return [item.embedding for item in resp.data]

    def generate_image_summary(self, images: List[bytes]) -> List[str]:
        """
        画像の要約を生成する。
        各画像をVision LLMで解析し、日本語で説明テキストを返す。
        """
        summaries: List[str] = []
        for image_bytes in images:
            # 画像をbase64エンコード
            b64 = base64.b64encode(image_bytes).decode("ascii")
            messages = [
                {"role": "system", "content": "あなたは日本語で画像を説明します。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "この画像を日本語で詳細に説明してください。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ]

            kwargs = self._build_completion_kwargs(
                model=self.image_model,
                messages=messages,
                temperature=0.3,
                api_key=self.image_model_api_key,
                base_url=self.image_llm_base_url,
                max_tokens=self.max_tokens_vision,
                timeout=self.image_timeout_seconds,
            )
            self._log_llm_send(kind="image_summary", payload=kwargs, model=self.image_model, stream=False)
            resp = litellm.completion(**kwargs)
            self._log_llm_recv(kind="image_summary", payload=self.response_to_dict(resp), model=self.image_model, stream=False)
            summaries.append(_first_choice_content(resp))

        return summaries

    def response_to_dict(self, resp: Any) -> Dict[str, Any]:
        """Responseオブジェクトをログ/デバッグ用のdictに変換する。"""
        return _response_to_dict(resp)

    def response_content(self, resp: Any) -> str:
        """Responseから本文（choices[0].message.content）を取り出す。"""
        return _first_choice_content(resp)

    def stream_text_deltas(
        self,
        resp_stream: Iterable[Any],
        *,
        tool_calls_state: Optional[dict[int, dict]] = None,
        kind: str = "chat",
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """ストリーミング応答から本文（delta.content）だけを逐次yieldする。

        /api/chat では本文をSSEで流したい一方で、同期メタJSONは tool call で回収する。
        そのため、tool call の断片は `tool_calls_state` に蓄積し、本文はyieldで返す。
        """
        collected: list[str] = []
        for chunk in resp_stream:
            if tool_calls_state is not None:
                self._accumulate_stream_tool_calls(tool_calls_state, chunk)
            delta = _delta_content(chunk)
            if delta:
                collected.append(delta)
                yield delta
        # ストリーム完了時に受信ログを出す
        full_text = "".join(collected)
        self._log_llm_recv(kind=kind, payload={"content": full_text}, model=model or self.model, stream=True)

    def parse_tool_call_arguments(self, tool_calls_state: dict[int, dict], *, tool_name: str) -> Optional[dict]:
        """蓄積した tool call から、指定ツールの arguments を JSON として取り出す。"""
        if not tool_calls_state:
            return None
        # index順で安定させる（最後に来たものを優先）
        for _idx in sorted(tool_calls_state.keys(), reverse=True):
            tc = tool_calls_state.get(_idx) or {}
            if str(tc.get("name") or "") != str(tool_name):
                continue
            args_text = str(tc.get("arguments") or "").strip()
            if not args_text:
                return None
            try:
                return json.loads(args_text)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    "tool call arguments parse failed",
                    extra={"tool_name": tool_name},
                    exc_info=exc,
                )
                self.logger.debug("tool call arguments(raw): %s", _truncate_for_log(args_text, self._DEBUG_PREVIEW_CHARS))
                return None
        return None

    @staticmethod
    def _accumulate_stream_tool_calls(tool_calls_state: dict[int, dict], chunk: Any) -> None:
        """streamの tool_calls 断片を結合して tool_calls_state へ蓄積する。"""

        def _to_dict(obj: Any) -> dict:
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:  # noqa: BLE001
                    return {}
            if hasattr(obj, "dict"):
                try:
                    return obj.dict()
                except Exception:  # noqa: BLE001
                    return {}
            try:
                return dict(obj)
            except Exception:  # noqa: BLE001
                return {}

        for tc_raw in _stream_delta_tool_calls(chunk):
            tc = _to_dict(tc_raw)
            try:
                idx = int(tc.get("index"))
            except Exception:  # noqa: BLE001
                continue

            entry = tool_calls_state.get(idx) or {"id": None, "name": None, "arguments": ""}
            if tc.get("id"):
                entry["id"] = tc.get("id")

            fn = _to_dict(tc.get("function"))
            if fn.get("name"):
                entry["name"] = fn.get("name")
            if fn.get("arguments"):
                entry["arguments"] = str(entry.get("arguments") or "") + str(fn.get("arguments") or "")

            tool_calls_state[idx] = entry
