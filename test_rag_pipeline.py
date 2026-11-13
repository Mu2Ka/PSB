"""Проверки без внешних API и платных запросов: python -m unittest -v."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import rag_pipeline as rag


def settings(root):
    return rag.Settings(
        base_url="https://proxy.test/v1", api_key="test-key", llm_model="platform-model",
        embedding_model="BAAI/bge-m3", reranker_model="Qwen/Qwen3-Reranker-8B",
        reranker_url="https://proxy.test/v1/rerank", knowledge_dir=Path(root),
        chroma_dir=Path(root) / "index", top_k=5, top_n=2,
        temperature=0.1, max_tokens=1000, timeout=10,
    )


class RecordingEmbeddings(Embeddings):
    calls = 0

    def embed_documents(self, texts):
        type(self).calls += len(texts)
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.5]


class PipelineTests(unittest.TestCase):
    def test_local_secret_survives_template_copy_and_blank_process_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("RAG_PROVIDER=auto\nOPENROUTER_API_KEY=\n", encoding="utf-8")
            (root / ".env.local").write_text("OPENROUTER_API_KEY=test-local-key\n", encoding="utf-8")
            with patch.object(rag, "ROOT", root), patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=True):
                config = rag.Settings.from_env()
                self.assertEqual(config.provider, "openrouter")
                self.assertEqual(config.api_key, "test-local-key")
                (root / ".env").write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")
                self.assertEqual(rag.Settings.from_env().api_key, "test-local-key")
                self.assertNotIn("test-local-key", (root / ".env").read_text(encoding="utf-8"))

    def test_nonempty_process_values_override_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("OPENROUTER_API_KEY=test-main-key\n", encoding="utf-8")
            (root / ".env.local").write_text("OPENROUTER_API_KEY=test-local-key\n", encoding="utf-8")
            with patch.object(rag, "ROOT", root), patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-process-key"}, clear=True):
                self.assertEqual(rag.Settings.from_env().api_key, "test-process-key")
                del os.environ["OPENROUTER_API_KEY"]
                self.assertEqual(rag.Settings.from_env().api_key, "test-local-key")
                # Чтение настроек не сохраняет ключи в глобальном окружении процесса.
                self.assertNotIn("OPENROUTER_API_KEY", os.environ)

    def test_environment_files_are_resolved_from_script_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env.local").write_text("OPENROUTER_API_KEY=test-root-key\n", encoding="utf-8-sig")
            with patch.object(rag, "ROOT", root), patch.dict(os.environ, {}, clear=True):
                self.assertNotEqual(Path.cwd(), root)
                self.assertEqual(rag.Settings.from_env().api_key, "test-root-key")

    def test_chunks_preserve_unicode_content_and_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            text = "Закон 🏛\r\n" * 131
            (Path(directory) / "закон.txt").write_bytes(text.encode("utf-8-sig"))
            chunks = rag.load_chunks(Path(directory))
            self.assertEqual("".join(d.page_content for d in chunks), text)
            self.assertTrue(all(len(d.page_content) == 500 for d in chunks[:-1]))
            for doc in chunks:
                self.assertEqual(doc.page_content, text[doc.metadata["start"]:doc.metadata["end"]])

    def test_actual_knowledge_base(self):
        directory = rag.ROOT / "knowledge-base"
        chunks = rag.load_chunks(directory)
        self.assertEqual(len({d.metadata["source"] for d in chunks}), 21)
        self.assertTrue(all(0 < len(d.page_content) <= 500 for d in chunks))
        self.assertIn("Документ:", rag.format_docs(chunks[:1]))

    def test_empty_knowledge_base_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "нет .txt"):
                rag.load_chunks(Path(directory))

    def test_environment_aliases_and_priority(self):
        env = {"OPENAI_BASE_URL": "https://proxy.test/v1/", "OPENAI_API_KEY": "test-key",
               "GEN_MODEL": "platform-model", "EMB_MODEL": "platform-bge",
               "RERANKER_MODEL": "platform-reranker"}
        with patch.dict(os.environ, env, clear=True), patch.object(rag, "read_environment", side_effect=lambda: dict(os.environ)):
            config = rag.Settings.from_env()
            self.assertEqual(config.llm_model, "platform-model")
            self.assertEqual(config.embedding_model, "platform-bge")
            self.assertEqual(config.reranker_url, "https://proxy.test/v1/rerank")
            os.environ["LLM_MODEL"] = "preferred-model"
            self.assertEqual(rag.Settings.from_env().llm_model, "preferred-model")

    def test_missing_credentials_report_names_only(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(rag, "read_environment", return_value={}):
            with self.assertRaisesRegex(ValueError, "LLM_BASE_URL, LLM_API_KEY, LLM_MODEL"):
                rag.Settings.from_env()

    def test_openrouter_uses_separate_models_without_platform_credentials(self):
        env = {"OPENROUTER_API_KEY": "test-router-key", "EMB_MODEL": "BAAI/bge-m3"}
        with patch.dict(os.environ, env, clear=True), patch.object(rag, "read_environment", side_effect=lambda: dict(os.environ)):
            config = rag.Settings.from_env()
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.api_key, "test-router-key")
        self.assertEqual(config.llm_model, "nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(config.embedding_model, "openai/text-embedding-3-small")

    def test_auto_prefers_platform_when_both_are_available(self):
        env = {"OPENROUTER_API_KEY": "test-router-key", "LLM_BASE_URL": "https://proxy.test/v1",
               "LLM_API_KEY": "test-platform-key", "LLM_MODEL": "platform-model"}
        with patch.dict(os.environ, env, clear=True), patch.object(rag, "read_environment", side_effect=lambda: dict(os.environ)):
            config = rag.Settings.from_env()
        self.assertEqual(config.provider, "platform")
        self.assertEqual(config.embedding_model, "BAAI/bge-m3")

    def test_hybrid_search_fuses_rankings_without_reranker_api(self):
        docs = [Document(page_content=text, metadata={"source": f"doc{i}", "chunk": 1,
                                                     "start": 0, "end": len(text)})
                for i, text in enumerate(["аренда северный склад", "налоги доход", "труд испытание"])]
        with patch.object(rag, "create_kb") as store, patch.object(rag, "load_chunks", return_value=docs), \
                patch.object(rag, "ChatOpenAI") as llm, patch.object(rag, "rerank") as api_rerank:
            store.return_value.as_retriever.return_value.invoke.return_value = [docs[1], docs[0]]
            assistant = rag.LegalRAG(replace(settings("."), provider="openrouter"))
            from unittest.mock import Mock
            assistant.chain = Mock()
            assistant.chain.invoke.return_value = "Ответ по документу [1]"
            answer, sources = assistant.ask("аренда северный склад")
        self.assertEqual(sources[0], docs[0])
        self.assertIn("[1]", answer)
        api_rerank.assert_not_called()

    def test_rerank_maps_scores_to_original_documents(self):
        docs = [Document(page_content=f"Норма {i}") for i in range(3)]

        def endpoint(request):
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 2, "relevance_score": 0.9}]})

        client = httpx.Client(transport=httpx.MockTransport(endpoint))
        with patch.object(rag.httpx, "Client", return_value=client):
            self.assertEqual(rag.rerank("вопрос", docs, settings(".")), [docs[2], docs[0]])

    def test_bad_rerank_index_is_rejected(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"results": [{"index": -1, "relevance_score": 1}]})))
        with patch.object(rag.httpx, "Client", return_value=client):
            with self.assertRaisesRegex(ValueError, "формат ответа"):
                rag.rerank("вопрос", [Document(page_content="норма")], settings("."))

    def test_empty_model_response_is_retried_without_repeating_retrieval(self):
        from unittest.mock import Mock
        assistant = rag.LegalRAG.__new__(rag.LegalRAG)
        assistant.settings = settings(".")
        assistant.retriever = Mock()
        assistant.chain = Mock()
        assistant.chain.invoke.side_effect = ["  ", "Подтверждённый ответ [1]"]
        doc = Document(page_content="Норма", metadata={"source": "закон.txt", "chunk": 1, "start": 0, "end": 5})
        with patch.object(rag, "rerank", return_value=[doc]):
            answer, sources = assistant.ask("Вопрос")
        self.assertEqual(answer, "Подтверждённый ответ [1]")
        self.assertEqual(sources, [doc])
        assistant.retriever.invoke.assert_called_once()
        self.assertEqual(assistant.chain.invoke.call_count, 2)

    def test_rerank_retries_transient_error(self):
        attempts = []

        def endpoint(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1}]})

        client = httpx.Client(transport=httpx.MockTransport(endpoint))
        with patch.object(rag.httpx, "Client", return_value=client), patch.object(rag.time, "sleep"):
            result = rag.rerank("вопрос", [Document(page_content="норма")], settings("."))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(result[0].page_content, "норма")

    def test_persistent_index_reused_and_invalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "закон.txt"
            source.write_text("Юридическая норма. " * 70, encoding="utf-8")
            config = settings(directory)
            RecordingEmbeddings.calls = 0
            with patch.object(rag, "OpenAIEmbeddings", return_value=RecordingEmbeddings()):
                store = rag.create_kb(config)
                count = len(store.get(include=[])["ids"])
                calls = RecordingEmbeddings.calls
                reused = rag.create_kb(config)
                self.assertEqual(len(reused.get(include=[])["ids"]), count)
                self.assertEqual(RecordingEmbeddings.calls, calls)
                self.assertTrue(reused.similarity_search("норма", k=1))
                source.write_text("Изменённая норма.", encoding="utf-8")
                changed = rag.create_kb(config)
                self.assertEqual(len(changed.get(include=[])["ids"]), 1)
                self.assertGreater(RecordingEmbeddings.calls, calls)
                # Закрываем SQLite/HNSW перед удалением TemporaryDirectory на Windows.
                store._client._system.stop()


if __name__ == "__main__":
    unittest.main()
