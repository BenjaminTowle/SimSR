import torch

from datasets import Dataset
from transformers import BertTokenizer
from typing import Optional, List

from src.agents.baseagent import BaseAgent, BaseAgentOutput
from src.modeling.biencoder import BertBiencoder, DistilBertBiencoder, DistilBertCVAE


class BiEncoderFAISSRetriever(BaseAgent):

    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290",
        model_type="bert",
        **kwargs
    ) -> None:

        if model_type == "bert":
            model = BertBiencoder.from_pretrained(model_path)
        elif model_type == "distilbert":
            model = DistilBertBiencoder.from_pretrained(model_path)
        elif model_type == "cvae":
            model = DistilBertCVAE.from_pretrained(model_path, z=kwargs["z"])
        else:
            raise ValueError("Model type not recognised")

        self.device = device
        self.model = model.to(device)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.index = self.build_index(dataset, device=device)

    def _build_index(self, samples, device: str = "cuda"):
        """Map fn"""
        input_ids = self.tokenizer(
            samples["responses"], max_length=32, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        samples["embeddings"] = self.model.forward_embedding(input_ids).cpu().numpy()
        return samples
        
    @torch.no_grad()
    def build_index(
        self, 
        dataset: Dataset,
        device: str = "cuda"
    ):

        dataset = dataset.map(lambda x: self._build_index(x, device=device), batched=True, batch_size=8)
        dataset.add_faiss_index(column="embeddings")

        return dataset

    @torch.no_grad()
    def act(
            self,
            query: str,
            k: int,
            docs: Optional[List[str]] = None,
    ):

        query_tokens = self.tokenizer([query], return_tensors="pt")

        query_embed = self.model.forward_embedding(
            query_tokens.input_ids.to(self.device))[0].cpu().numpy()

        outputs = self.index.search("embeddings", query_embed, k=len(self.index))
        scores = outputs.scores

        scores, retrieved_examples = self.index.get_nearest_examples("embeddings", query_embed, k=k)
        docs = retrieved_examples["responses"]
       
        return BaseAgentOutput(
            docs=docs,
            doc_scores=scores,
            doc_embeds=retrieved_examples["embeddings"],
            query_embed=query_embed
        )


class MCVAERetriever(BaseAgent):
    """
    Re-implementation of https://arxiv.org/abs/1903.10630.
    """
    def __init__(
        self,
        dataset: Dataset,
        device="cuda",
        model_path="biencoder/checkpoint-49290",
        model_type="none",
        n: int = 15,
        s: int = 300,
        **kwargs
        
    ) -> None:

        self.model = BiEncoderFAISSRetriever(
            dataset=dataset,
            device=device,
            model_path=model_path,
            model_type="cvae",
            **kwargs
        )
    
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = device
        self.n = n
        self.s = s

    @torch.no_grad()
    def act(
            self,
            query: str,
            k: int,
            docs: Optional[List[str]] = None,
    ):

        # Retrieve longlist
        outputs = self.model.act(query, k=self.n)

        # Generate context vectors
        embeds = self.model.model.generate_embedding(
            torch.tensor(outputs.query_embed).to(self.device), num_samples=self.s
        )

        scores = torch.matmul(
            embeds, torch.tensor(outputs.doc_embeds).to(self.device).T
        )

        votes = scores.argmax(-1).cpu().numpy().tolist()
        vote_sums = [(i, votes.count(i)) for i in range(self.n)]
        vote_sums = list(reversed(sorted(vote_sums, key=lambda x: x[1])))[:k]
        vote_idxs, _ = zip(*vote_sums)
        docs = [outputs.docs[idx] for idx in vote_idxs]

        return BaseAgentOutput(docs=docs)
