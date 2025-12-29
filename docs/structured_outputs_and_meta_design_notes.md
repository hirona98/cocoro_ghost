# Structured Outputs 導入と /api/chat メタ生成（設計前まとめ）

作成日: 2025-12-29  
対象リポジトリ: `cocoro_ghost`

このファイルは、今回の会話で確定した要求・制約・判断理由を、これから設計/実装に入る前の「合意メモ」としてまとめる。

## 背景（現状）

- LLM呼び出しは `litellm.completion()` を中心にしており、OpenAI `chat.completions` 互換の `messages` / `choices[0].message.content` / `delta.content` 前提が各所にある（`cocoro_ghost/llm_client.py`、`cocoro_ghost/memory.py` のSSE）。
- JSON生成は `response_format={"type":"json_object"}` + 文字列修復（コードフェンス除去、末尾カンマ除去、制御文字エスケープ等）で「なんとか `json.loads` する」設計になっている（`cocoro_ghost/llm_client.py`）。
- `/api/chat` は本文ストリームと同一のLLM呼び出しで、本文末尾に内部JSON（partner_affect trailer）を混在させて回収している（仕様: `docs/prompts.md`、実装: `cocoro_ghost/memory.py`）。

## 主目的（今回のゴール）

- **Structured Outputs を使って、プロジェクト内の「JSONとして扱う出力」を全て綺麗にする。**
  - reflection / entity / fact / loop / summaries / names_only 等、LLMがJSONを返す系の全てが対象。
  - 「修復して通す」のではなく、**最初から型・キー・構造が保証される**状態にする。

## 重要な要求 / 制約

- `/api/chat` のメタ（反射/affect 等）は **同期で欲しい**（同ターン保存・同ターン反映が必要）。
- 追加のLLM呼び出し（2回呼び）は **レイテンシが許容できない**。
- クライアント（別アプリ）は **本文しか使わない**（メタJSONはクライアント必須ではない）。
- 運用前のため、**後方互換やマイグレーションは不要**（既存クライアント互換のための温存より、あるべき姿へ寄せる）。
- シンプルさを優先する。

## 途中で検討した案と判断理由

### 案A: Structured Outputsだけ先に入れ、Responses APIは後回し

- **Structured OutputsはJSON品質に直結**し、現状の不安定さ（修復・`json.loads`前提）を減らせる。
- Responses API化は、ストリーミングやレスポンス解釈の前提が変わり改修範囲が広い。
  - 現状は `delta.content` 前提だが、Responsesは `response.output_text.delta` 等のイベント前提になり、`memory.py` のSSE実装を大きく触る必要がある。
- したがって「主目的＝JSON安定化」だけなら、Responses APIは必須ではない。

=> ただし、`/api/chat` でメタを同一ターン同期で取りたい要件があり、**1回呼びで本文+メタを両立する設計**が別途必要になった。

### 案B: Structured Outputsで「reply_text + meta」を単一JSONとして返す

- 技術的には可能（出力全体を JSON に強制）。
- しかし `/api/chat` は本文をSSEでストリームしており、単一JSON方式にすると:
  - 返信が最後まで完成するまで UI に本文を流せない（体験が悪い）
  - もしくは JSONストリームを壊さずに `reply_text` だけ逐次取り出す専用パーサが必要になり、シンプルさに反する

=> 「本文ストリーム」を維持する前提では不採用。

### 案C: 2回呼び（本文ストリーム中にメタ生成を並列）

- ストリーム中に2回目を投げること自体は可能（別スレッド/別HTTP接続で実行）。
- ただし整合性（最終本文に依存したメタ）を守るなら、本文が確定するまで2回目を開始できず、完了レイテンシは短くならない。
- さらに、2回呼びそのものが「追加レイテンシ許容できない」という制約に反する。

=> 不採用。

### 案D: 1回呼びで本文ストリーム + メタ厳格JSON（tool call / function calling）

- 同一リクエスト内で、本文は通常テキストとしてストリームし続けられる。
- 同一リクエスト内で、メタは「ツール呼び出し」として **strictなJSON Schema** で回収できる（Structured Outputs相当の保証を得られる）。
- 追加のLLM呼び出しが不要なので、レイテンシも増えない（モデルがツール呼び出しを含めて生成する分は同一ストリームの範囲）。

=> **採用方針**（これからの設計のベース）。

## 決定事項（現時点での合意）

- `/api/chat` は **1回呼び**で、本文ストリームを維持したまま、同期でメタ（反射/affect等）も回収する。
- 「本文末尾に内部JSONを混在させる（partner_affect trailer）」方式は、あるべき姿ではないため **廃止方向**。
- JSON出力は全面的に Structured Outputs（厳格スキーマ）へ寄せる。
- Responses API は「今すぐ必須」ではない。必要が出たら後で検討する。

## これから設計で詰めること（未決）

### 1) /api/chat のメタ範囲

- 同期で回収するメタは、最低限「partner_affect / topic_tags / salience / confidence / response_policy」相当でよいか。
- entity/fact/loop のような重い抽出まで「同期」に含めるか（含めるとモデル負荷が増え、本文生成にも影響しやすい）。

### 2) SSEイベント仕様（サーバ内の都合 vs クライアント無関心）

- クライアントは本文のみ使用するため、SSEは現行 `token` を維持しつつ、
  - サーバ側だけが利用する `meta` イベントを追加する（クライアントは無視可能）
  - あるいは SSEには流さず、サーバ内部で保存だけして `done` に含める
 どちらが「本来あるべき姿」かを決める。

### 3) スキーマの一元管理

- `docs/prompts.md` の各JSON例を、Pydanticモデル（またはJSON Schema定義）としてコード側で一元化する。
- 「プロンプト文」と「スキーマ」を分離し、実装で常にスキーマを正とする。

## 影響範囲（設計/実装の主要ポイント）

- `cocoro_ghost/llm_client.py`
  - JSON系を “文字列修復→json.loads” から “スキーマで厳格に取得” に置換する核。
  - streaming時の tool call / function call の取り回し（イベント抽出）が必要になる可能性が高い。
- `cocoro_ghost/memory.py`
  - `/api/chat` の partner_affect trailer 回収ロジックを撤去し、新方式（tool call等）へ。
  - SSEのイベント設計（token / meta / done）を整理。
- `cocoro_ghost/worker.py`, `cocoro_ghost/memory_pack_builder.py`
  - `json.loads(llm_client.response_content(resp))` 前提箇所を置換し、構造化結果を直接受け取る形へ。
- `docs/prompts.md`, `docs/prompt_usage_map.md`
  - chatトレーラー方式の記述を更新し、Structured Outputs/スキーマ主導に合わせて修正する。

## まとめ（次アクション）

設計の中心は「**1回呼びのまま本文ストリームを維持し、メタJSONだけを厳格スキーマで同一リクエストから回収する**」。
そのために、tool call / function calling を使った Structured Outputs 相当の実装を前提に詳細設計へ進む。

