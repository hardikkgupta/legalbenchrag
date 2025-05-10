from typing import Literal

from legalbenchrag.methods.baseline import ChunkingStrategy, RetrievalStrategy
from legalbenchrag.utils.ai import AIEmbeddingModel, AIRerankModel

chunk_strategy_names: list[Literal["naive", "rcts", "semantic"]] = ["naive", "rcts", "semantic"]
rerank_models: list[AIRerankModel | None] = [
    None,
    AIRerankModel(company="cohere", model="rerank-english-v3.0"),
]
chunk_sizes: list[int] = [500, 1000]  # Added larger chunk size for semantic strategy
top_ks: list[int] = [1, 2, 4, 8, 16, 32, 64]

RETRIEVAL_STRATEGIES: list[RetrievalStrategy] = []
for chunk_strategy_name in chunk_strategy_names:
    for chunk_size in chunk_sizes:
        chunking_strategy = ChunkingStrategy(
            strategy_name=chunk_strategy_name,
            chunk_size=chunk_size,
        )
        for rerank_model in rerank_models:
            for top_k in top_ks:
                # For semantic strategy, use larger embedding_topk to get more candidates for reranking
                embedding_topk = 300 if rerank_model is not None else top_k
                if chunk_strategy_name == "semantic":
                    embedding_topk = max(embedding_topk, 100)  # Ensure we get enough candidates for semantic chunks
                
                RETRIEVAL_STRATEGIES.append(
                    RetrievalStrategy(
                        chunking_strategy=chunking_strategy,
                        embedding_model=AIEmbeddingModel(
                            company="openai",
                            model="text-embedding-3-large",
                        ),
                        embedding_topk=embedding_topk,
                        rerank_model=rerank_model,
                        rerank_topk=top_k,
                        token_limit=None,
                    ),
                )
