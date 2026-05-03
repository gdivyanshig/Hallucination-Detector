import nltk
from nltk.tokenize import sent_tokenize
import spacy
import wikipedia
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

@dataclass
class VerificationResult:
    claim: str
    evidence: str
    label: str
    confidence: float
    trust_score: float


class HallucinationDetector:
    def __init__(self):
        print("Initializing lightweight detector...")
        self.nlp = spacy.blank("en")  # lightweight tokenizer
        self.vectorizer = TfidfVectorizer()

    # ---------------- EXTRA CHECKS ----------------
    def is_math_expression(self, text: str) -> bool:
        return any(op in text for op in ["+", "-", "*", "/", "="])

    def is_generic_claim(self, claim: str) -> bool:
        claim = claim.lower()

        generic_subjects = ["dogs", "cats", "animals", "people", "things"]
        generic_patterns = ["are", "is", "can be", "usually", "generally"]

        words = claim.split()

        # detect generic subject + generic statement
        if any(sub in words for sub in generic_subjects):
            if any(pattern in claim for pattern in generic_patterns):
                return True

        return False
    
    def is_reversed_claim(self, claim: str, evidence: str) -> bool:
        claim = claim.lower()
        evidence = evidence.lower()

        if " in " in claim:
            parts = claim.split(" in ")
            if len(parts) == 2:
                subject = parts[0].strip()
                obj = parts[1].strip()

                # If evidence says "obj is in subject", then claim is reversed
                if f"{obj} is in {subject}" in evidence:
                    return True

        return False

    # ---------------- STAGE 1 ----------------
    def stage1_atomic_claim_extraction(self, text: str) -> List[str]:
        # Better splitting (handles multi-line input properly)
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        claims = []

        for sent in sentences:
            if '?' in sent or '!' in sent:
                continue
            if len(sent.strip()) > 0:
                claims.append(sent.strip())

        return claims

    # ---------------- STAGE 2 ----------------
    def stage2_evidence_retrieval(self, claim: str, top_k: int = 3) -> List[str]:
        try:
            search_results = wikipedia.search(claim, results=top_k * 2)
        except:
            return []

        evidence_snippets = []

        # Remove common words
        stop_words = {"is", "in", "the", "of", "and", "a", "an", "on", "at", "to"}
        claim_words = set([w for w in claim.lower().split() if w not in stop_words])

        for title in search_results:
            try:
                page = wikipedia.page(title)
                summary = page.summary

                # Extract relevant sentences
                sentences = summary.split('.')
                relevant_sentences = []

                for s in sentences:
                    if any(word in s.lower() for word in claim_words):
                        relevant_sentences.append(s)

                if relevant_sentences:
                    evidence_snippets.append('. '.join(relevant_sentences[:2]))

            except:
                continue

            if len(evidence_snippets) >= top_k:
                break

        return evidence_snippets

    # ---------------- STAGE 3 ----------------
    def stage3_nli_verification(self, claim: str, evidence: List[str]) -> Tuple[str, float, float]: # tf-idf verification
        if not evidence:
            return 'ABSTAINED❌', 0.0, 0.0

        texts = [claim] + evidence
        tfidf = self.vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        max_sim = float(np.max(similarities))
        best_index = int(np.argmax(similarities))
        best_evidence = evidence[best_index]

        # Keyword overlap
        claim_words = set(claim.lower().split())
        evidence_words = set(best_evidence.lower().split())
        overlap_score = len(claim_words & evidence_words) / max(len(claim_words), 1)

        # Entity match (simple)
        claim_entities = [word for word in claim.lower().split() if len(word) > 3]
        evidence_entities = best_evidence.lower().split()
        entity_match = any(word in evidence_entities for word in claim_entities)

        # Final decision (tuned thresholds)
        if max_sim > 0.3 and overlap_score > 0.3 and entity_match:
            if self.is_reversed_claim(claim, best_evidence):
                return 'ABSTAINED❌', max_sim, 0.0
            return 'VERIFIED✅', max_sim, max_sim

        elif max_sim < 0.15:
            return 'HALLUCINATED🟡', max_sim, 0.0

        else:
            return 'ABSTAINED❌', max_sim, 0.0

    # ---------------- STAGE 4 ----------------
    def stage4_hybrid_switch(self, text: str) -> str:
        text = text.lower()
        if "why" in text or "how" in text:
            return "reasoning query"
        return "factual query"

    # ---------------- MAIN ----------------
    def detect_hallucinations(self, llm_output: str) -> Dict[str, Any]:
        route = self.stage4_hybrid_switch(llm_output)
        print(f"Routing to: {route}")

        claims = self.stage1_atomic_claim_extraction(llm_output)

        results = []
        verified, hallucinated, abstained = 0, 0, 0

        for claim in claims:

            # ✅ 1. Math 
            if self.is_math_expression(claim):
                try:
                    left, right = claim.split("=")

                    if eval(left.strip()) == eval(right.strip()):
                        results.append(VerificationResult(
                            claim=claim,
                            evidence="Mathematical evaluation",
                            label="VERIFIED✅",
                            confidence=1.0,
                            trust_score=1.0
                        ))
                        verified += 1
                    else:
                        results.append(VerificationResult(
                            claim=claim,
                            evidence="Incorrect mathematical statement",
                            label="ABSTAINED❌",
                            confidence=1.0,
                            trust_score=0.0
                        ))
                        abstained += 1

                    continue

                except:
                     pass

            # ✅ 2. Generic → HALLUCINATED
            if self.is_generic_claim(claim):
                results.append(VerificationResult(
                    claim=claim,
                    evidence="Generic statement",
                    label="HALLUCINATED🟡",
                    confidence=0.0,
                    trust_score=0.0
                ))
                hallucinated += 1
                continue

            # ✅ 3. Normal pipeline
            evidence = self.stage2_evidence_retrieval(claim)
            label, conf, trust = self.stage3_nli_verification(claim, evidence)

            results.append(VerificationResult(
                claim=claim,
                evidence='; '.join(evidence[:1]) if evidence else "No evidence",
                label=label,
                confidence=conf,
                trust_score=trust
            ))

            if label == 'VERIFIED✅':
                verified += 1
            elif label == 'HALLUCINATED🟡':
                hallucinated += 1
            else:
                abstained += 1

        total = len(claims)

        metrics = {
            'accuracy': verified / total if total else 0,
            'hallucination_rate': hallucinated / total if total else 0,
            'abstention_rate': abstained / total if total else 0,
            'trust_score': (verified / total * 100) if total else 0
        }

        return {
            'results': results,
            'metrics': metrics,
            'route': route
        }


# ---------------- RUN ----------------
if __name__ == "__main__":
    detector = HallucinationDetector()

    print("\nEnter your text (press Enter twice to finish):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    llm_output = "\n".join(lines)

    output = detector.detect_hallucinations(llm_output)

    print("\nDetailed Results:")
    for i, r in enumerate(output['results'], 1):
        print(f"{i}. {r.claim}")
        print(f"   → {r.label} (conf: {r.confidence:.2f})")
        print(f"   → Evidence: {r.evidence[:100]}...\n")
