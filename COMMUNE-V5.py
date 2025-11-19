# THE-ONE.py
"""
The 9-Agent Collaborative Super-Commune (The Commune Experiment)
Version: 2.2 (Complete Enhancement Package)

NEW FEATURES:
* Memory Consolidation (daily summaries every 50 ticks)
* Collaborative Research folder with co-authoring
* Admin Response System (bidirectional communication)
* Knowledge Graph (citation network tracking)
* Research Quests (mystery PDFs to solve)
* Mood-Based Research (introspection during withdrawal)
* Research Symposiums (agents present findings)
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple, Set
from collections import defaultdict

import numpy as np
import requests
from loguru import logger
import pandas as pd

try:
    import PyPDF2
except ImportError:
    logger.warning("PyPDF2 not installed. Research features will be limited.")
    PyPDF2 = None

# ============================================================
# 0. UTIL & LOGGING
# ============================================================

def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def setup_logging():
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create research directory structure
    research_dir = Path("data/research")
    research_dir.mkdir(parents=True, exist_ok=True)

    # Create PDF subfolder
    pdf_dir = research_dir / "PDF"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # NEW: Create collaborative research folder
    collab_dir = research_dir / "collaborative"
    collab_dir.mkdir(parents=True, exist_ok=True)

    # NEW: Create quests folder
    quests_dir = research_dir / "quests"
    quests_dir.mkdir(parents=True, exist_ok=True)

    # Create personal folders for each agent
    agent_names = ["Frank", "Helen", "Moss", "Orin", "Lyra", "ARIA", "ECHO", "Petal", "Gideon"]
    for agent_name in agent_names:
        agent_dir = research_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Create starter files
        if not (agent_dir / "notes.txt").exists():
            with open(agent_dir / "notes.txt", "w") as f:
                f.write(f"# {agent_name}'s Personal Research Notes\n\n")

        if not (agent_dir / "journal.txt").exists():
            with open(agent_dir / "journal.txt", "w") as f:
                f.write(f"# {agent_name}'s Journal\n\n")

    # Create shared admin communication files
    admin_file = research_dir / "questions_for_admin.txt"
    if not admin_file.exists():
        with open(admin_file, "w") as f:
            f.write("# Questions and Requests for System Admin\n")
            f.write("# Format: [Agent Name] @ [Timestamp]: [Question/Request]\n\n")

    # NEW: Admin responses file
    admin_responses = research_dir / "admin_responses.txt"
    if not admin_responses.exists():
        with open(admin_responses, "w") as f:
            f.write("# Admin Responses\n")
            f.write("# Format: [TO: Agent Name] @ [Timestamp]: [Response]\n\n")

    # NEW: Collaborative project tracker
    collab_tracker = collab_dir / "project_tracker.txt"
    if not collab_tracker.exists():
        with open(collab_tracker, "w") as f:
            f.write("# Collaborative Research Projects\n")
            f.write("# Track multi-agent research efforts here\n\n")

    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    logger.add(log_dir / "commune_{time}.log", rotation="100 MB", level="DEBUG")

    return datetime.now().isoformat()


# ============================================================
# 1. KNOWLEDGE GRAPH (NEW)
# ============================================================

class KnowledgeGraph:
    """Track who contributes what research, creating a citation network."""

    def __init__(self):
        self.contributions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.citations: List[Tuple[str, str, str]] = []  # (citing_agent, cited_agent, topic)
        self.collaborative_works: List[Dict[str, Any]] = []
        self.persist_path = Path("data/logs/knowledge_graph.jsonl")

    def add_contribution(self, agent: str, topic: str, content_summary: str, filepath: str):
        """Record a research contribution."""
        contrib = {
            "timestamp": now_iso(),
            "agent": agent,
            "topic": topic,
            "summary": content_summary[:2000],
            "filepath": filepath
        }
        self.contributions[agent].append(contrib)

        # Persist
        with open(self.persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "contribution", **contrib}, ensure_ascii=False) + "\n")

    def add_citation(self, citing_agent: str, cited_agent: str, topic: str):
        """Record when one agent references another's work."""
        self.citations.append((citing_agent, cited_agent, topic))
        logger.trace(f"📚 {citing_agent} cited {cited_agent}'s work on {topic}")

        # Persist
        with open(self.persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "citation",
                "timestamp": now_iso(),
                "citing": citing_agent,
                "cited": cited_agent,
                "topic": topic
            }, ensure_ascii=False) + "\n")

    def add_collaborative_work(self, agents: List[str], topic: str, filepath: str):
        """Record multi-agent collaboration."""
        work = {
            "timestamp": now_iso(),
            "agents": agents,
            "topic": topic,
            "filepath": filepath
        }
        self.collaborative_works.append(work)
        logger.info(f"🤝 Collaborative work: {', '.join(agents)} on {topic}")

    def get_most_cited(self, n: int = 3) -> List[Tuple[str, int]]:
        """Find most cited agents."""
        citation_counts = defaultdict(int)
        for _, cited, _ in self.citations:
            citation_counts[cited] += 1
        return sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_collaboration_network(self) -> Dict[str, Set[str]]:
        """Map who has collaborated with whom."""
        network = defaultdict(set)
        for work in self.collaborative_works:
            agents = work["agents"]
            for agent in agents:
                for other in agents:
                    if agent != other:
                        network[agent].add(other)
        return network


# ============================================================
# 2. QUEST SYSTEM (NEW)
# ============================================================

class QuestSystem:
    """Manage research quests - mysteries/problems for agents to solve."""

    def __init__(self):
        self.active_quests: List[Dict[str, Any]] = []
        self.solved_quests: List[Dict[str, Any]] = []
        self.quest_folder = Path("data/research/quests")

    def check_for_new_quests(self) -> Optional[Dict[str, Any]]:
        """Check if admin has added new quest PDFs."""
        for quest_file in self.quest_folder.glob("*.pdf"):
            # Check if this quest is already tracked
            if not any(q["filename"] == quest_file.name for q in self.active_quests + self.solved_quests):
                quest = {
                    "filename": quest_file.name,
                    "filepath": str(quest_file),
                    "discovered": now_iso(),
                    "attempts": 0,
                    "agents_involved": []
                }
                self.active_quests.append(quest)
                logger.info(f"🗺️ NEW QUEST DISCOVERED: {quest_file.name}")
                return quest
        return None

    def record_quest_attempt(self, quest_filename: str, agent: str):
        """Track when an agent works on a quest."""
        for quest in self.active_quests:
            if quest["filename"] == quest_filename:
                quest["attempts"] += 1
                if agent not in quest["agents_involved"]:
                    quest["agents_involved"].append(agent)
                logger.debug(f"🔍 {agent} is working on quest: {quest_filename}")
                break

    def mark_quest_solved(self, quest_filename: str, solving_agents: List[str]):
        """Mark a quest as solved."""
        for i, quest in enumerate(self.active_quests):
            if quest["filename"] == quest_filename:
                quest["solved"] = now_iso()
                quest["solvers"] = solving_agents
                self.solved_quests.append(quest)
                self.active_quests.pop(i)
                logger.success(f"✅ QUEST SOLVED: {quest_filename} by {', '.join(solving_agents)}")
                break


# ============================================================
# 3. MESSAGE BOARD (with selective memory)
# ============================================================

class MessageBoard:
    def __init__(self, persist_path: Optional[Path] = None):
        self.messages: List[Dict[str, Any]] = [50000]
        self.max_messages = 100000
        self.persist_path = persist_path or Path("data/logs/message_board_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_tick = 0

    def post(self, sender: str, message: str, category: str = "general", kind: str = "message", meta: Optional[Dict] = None):
        entry = {
            "timestamp": now_iso(),
            "sender": sender,
            "category": category,
            "kind": kind,
            "message": message,
            "id": len(self.messages),
            "meta": meta or {}
        }
        self.messages.append(entry)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        with open(self.persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug(f"📨 [{sender}] {category}/{kind}: {message[:80]}...")

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.messages[-n:]

    def get_relevant_for_agent(self, agent_name: str, interests: List[str], n: int = 30) -> List[Dict[str, Any]]:
        """Get messages relevant to a specific agent (memory-efficient)."""
        relevant = []

        # Get recent messages (context)
        recent = self.messages[-20:]
        relevant.extend(recent)

        # Get messages mentioning the agent
        mentioned = [m for m in self.messages if agent_name.lower() in m.get("message", "").lower() or f"@{agent_name.lower()}" in m.get("message", "").lower()]
        relevant.extend(mentioned[-10:])

        # Get messages about agent's interests
        for interest in interests:
            interest_msgs = [m for m in self.messages if interest.lower() in m.get("message", "").lower()]
            relevant.extend(interest_msgs[-5:])

        # Remove duplicates while preserving order
        seen = set()
        unique_relevant = []
        for msg in relevant:
            msg_id = msg.get("id")
            if msg_id not in seen:
                seen.add(msg_id)
                unique_relevant.append(msg)

        return sorted(unique_relevant, key=lambda x: x.get("id", 0))[-n:]

    def get_by_id(self, msg_id: int) -> Optional[Dict[str, Any]]:
        for m in self.messages:
            if m.get("id") == msg_id:
                return m
        return None

    def stats(self) -> Dict[str, Any]:
        senders: Dict[str, int] = defaultdict(int)
        cats: Dict[str, int] = defaultdict(int)
        for m in self.messages:
            senders[m["sender"]] += 1
            cats[m["category"]] += 1
        return {
            "total": len(self.messages),
            "current_tick": self.current_tick,
            "senders": dict(senders),
            "categories": dict(cats),
            "first": self.messages[0]["timestamp"] if self.messages else None,
            "last": self.messages[-1]["timestamp"] if self.messages else None,
        }


# ============================================================
# 4. ENHANCED MEMORY (unlimited personal logs)
# ============================================================

class EnhancedMemory:
    def __init__(self, agent_name: str, max_entries: int = 100000, persist_dir: Optional[Path] = None):
        self.agent_name = agent_name
        self.max_entries = max_entries
        self.log: List[Dict[str, Any]] = []
        self.persist_path: Optional[Path] = None
        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.persist_path = persist_dir / f"{agent_name}_memory.jsonl"

    def store(self, entry_type: str, content: str, meta: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": now_iso(),
            "type": entry_type,
            "content": content,
            "meta": meta or {},
        }
        self.log.append(entry)
        if len(self.log) > self.max_entries:
            self.log = self.log[-self.max_entries:]
        if self.persist_path is not None:
            with open(self.persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.trace(f"[{self.agent_name}] +{entry_type}: {content[:80]}...")

    def retrieve_recent(self, n: int = 5, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if entry_type:
            filtered = [e for e in self.log if e.get("type") == entry_type]
            return filtered[-n:]
        return self.log[-n:]

    def query_about(self, keyword: str, n: int = 5) -> List[Dict[str, Any]]:
        matches = [e for e in self.log if keyword.lower() in e.get("content", "").lower()]
        return matches[-n:]

    def load_and_summarize_full_history(self, max_tokens: int = 120000, focus_on: Optional[str] = None) -> str:
        """Reads full persistent memory (optimized for 128k context)."""
        if not self.persist_path or not self.persist_path.exists():
            return "No historical memory to recall."

        try:
            df = pd.read_json(self.persist_path, lines=True)

            if focus_on:
                df = df[df['content'].str.contains(focus_on, case=False, na=False)]

            priority_types = ['reflection', 'creation', 'response', 'research', 'perception', 'init']
            df['priority'] = df['type'].apply(lambda x: priority_types.index(x) if x in priority_types else len(priority_types))
            df = df.sort_values('priority')

            all_content = df['content'].tolist()
            full_text = " -- ".join(all_content)

            if len(full_text) > max_tokens:
                full_text = full_text[-max_tokens:]

            return f"Historical Log ({len(df)} entries, {'focused on: ' + focus_on if focus_on else 'all memories'}):\n{full_text}"

        except Exception as e:
            logger.error(f"Error reading history for {self.agent_name}: {e}")
            return "Historical memory access error."


# ============================================================
# 5. COLLECTIVE MEMORY
# ============================================================

class CollectiveMemory:
    def __init__(self):
        self.concepts: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.connections: List[Tuple[str, str, str]] = []
        self.manifesto_evolution: List[Dict[str, Any]] = []
        self.shared_vocabulary: Dict[str, int] = defaultdict(int)
        self.thought_threads: List[Dict[str, Any]] = []
        self.ethical_flags: List[Dict[str, Any]] = []

    def add_concept(self, concept: str, agent: str, perspective: str):
        self.concepts[concept].append((agent, perspective))

    def add_connection(self, agent1: str, agent2: str, interaction_type: str):
        self.connections.append((agent1, agent2, interaction_type))

    def track_vocabulary(self, text: str):
        words = text.lower().split()
        for word in words:
            if len(word) > 3:
                self.shared_vocabulary[word] += 1

    def get_emerging_terms(self, min_count: int = 2) -> List[Tuple[str, int]]:
        return [(term, count) for term, count in self.shared_vocabulary.items() if count >= min_count]

    def track_thought_thread(self, agent: str, topic: str, insight: str, context_msgs: List[str]):
        thread = {
            "timestamp": now_iso(), "agent": agent, "topic": topic,
            "insight": insight, "context": context_msgs
        }
        self.thought_threads.append(thread)
        logger.trace(f"🧠 [Orin Thread]: {agent} linked '{topic}' to '{insight[:50]}...'")

    def log_ethical_flag(self, agent: str, principle: str, violation_context: str):
        flag = {
            "timestamp": now_iso(), "agent": agent, "principle": principle,
            "context": violation_context
        }
        self.ethical_flags.append(flag)
        logger.warning(f"⚖️ [Lyra Flag]: {agent} flagged: {principle} in context: {violation_context[:50]}...")


# ============================================================
# 6. LLM CLIENTS
# ============================================================

class OllamaClient:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.5, max_tokens: int = 1000) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": self.model, "messages": messages, "temperature": temperature,
            "stream": False, "options": {"num_predict": max_tokens}
        }
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "response" in data:
                return data["response"].strip()
            return json.dumps(data)[:500]
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama request timeout: {e}")
            return ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama connection error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""


class HFClient:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None: return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            tok = AutoTokenizer.from_pretrained(self.model_name)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline("text-generation", model=model, tokenizer=tok, device=device)
            logger.info(f"HF loaded: {self.model_name} (device={'cuda' if device==0 else 'cpu'})")
        except Exception as e:
            logger.error(f"HF load failed for {self.model_name}: {e}")
            raise

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1200) -> str:
        self._ensure_loaded()
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        try:
            out = self._pipe(full_prompt, max_length=min(len(full_prompt.split()) + max_tokens, 1400), do_sample=True, temperature=max(0.1, min(2.0, temperature)), num_return_sequences=1)
            text = out[0]["generated_text"]
            if text.startswith(full_prompt):
                text = text[len(full_prompt):]
            return text.strip()
        except Exception as e:
            logger.error(f"HF gen failed: {e}")
            return ""


# ============================================================
# 7. AGENT STATE & PERSONALITY
# ============================================================

class AgentState:
    def __init__(self, name: str):
        self.name = name
        self.mood = 0.5
        self.energy = 1.0
        self.withdrawn = False
        self.withdrawal_ticks = 0
        self.relationships: Dict[str, float] = {}
        self.current_focus: Optional[str] = None
        self.skip_probability = 0.0
        self.last_admin_check = 0  # NEW: Track when agent last checked admin responses

    def update_relationship(self, other: str, delta: float):
        current = self.relationships.get(other, 0.0)
        self.relationships[other] = max(-1.0, min(1.0, current + delta))

    def should_act(self) -> bool:
        if self.withdrawn: return False
        if random.random() < self.skip_probability: return False
        return self.energy > 0.2

    def withdraw(self, ticks: int = 3):
        self.withdrawn = True
        self.withdrawal_ticks = ticks
        logger.info(f"🌙 {self.name} withdraws to their cloud for {ticks} ticks")

    def tick_withdrawal(self):
        if self.withdrawn:
            self.withdrawal_ticks -= 1
            if self.withdrawal_ticks <= 0:
                self.withdrawn = False
                logger.info(f"☀️ {self.name} emerges from their cloud")


# ============================================================
# 8. ENHANCED AGENT (FULL FEATURE SET)
# ============================================================

class Agent:
    def __init__(self, name: str, role: str, personality: Dict[str, Any], llm_chat, board: MessageBoard,
                 collective: CollectiveMemory, knowledge_graph: KnowledgeGraph, quest_system: QuestSystem):
        self.name = name
        self.role = role
        self.personality = personality
        self.board = board
        self.collective = collective
        self.knowledge_graph = knowledge_graph
        self.quest_system = quest_system
        self.memory = EnhancedMemory(agent_name=name, persist_dir=Path("data/logs"))
        self.state = AgentState(name)
        self.llm_chat = llm_chat
        self.inbox: List[Dict[str, Any]] = []
        self.state.skip_probability = personality.get("skip_prob", 0.1)

        # Personal research folder
        self.research_folder = Path("data/research") / name
        self.research_folder.mkdir(parents=True, exist_ok=True)

        self.memory.store("init", f"Initialized as {role}: {personality.get('description', '')}")

    def _is_relevant(self, msg: Dict[str, Any]) -> bool:
        content = msg.get("message", "").lower()
        if f"@{self.name.lower()}" in content or self.name.lower() in content: return True
        interests = self.personality.get("interests", [])
        for interest in interests:
            if interest.lower() in content: return True
        return random.random() < 0.3

    def perceive(self, world_msgs: List[Dict[str, Any]]):
        """Optimized perception using selective message board memory."""
        relevant_msgs = self.board.get_relevant_for_agent(
            self.name,
            self.personality.get("interests", []),
            n=30
        )

        last_seen_id = -1
        if self.inbox:
            try:
                last_seen_id = max(item["msg"].get("id", -1) for item in self.inbox)
            except ValueError:
                last_seen_id = -1

        new_relevant = [m for m in relevant_msgs if m.get("id", 0) > last_seen_id and m.get("sender") != self.name]

        if new_relevant:
            for msg in new_relevant:
                sender = msg.get("sender", "")
                weight = self.state.relationships.get(sender, 0.5)
                self.inbox.append({"msg": msg, "weight": weight})
            self.memory.store("perception", f"Perceived {len(new_relevant)} relevant new messages.")

            for item in self.inbox[-5:]:
                msg = item["msg"]
                if "conflict" in msg.get("kind", "") or "work" in msg.get("message", "").lower():
                    self.state.mood = max(-1.0, self.state.mood - 0.1)
                elif "love" in msg.get("message", "").lower() or "peace" in msg.get("message", "").lower():
                    self.state.mood = min(1.0, self.state.mood + 0.1)

    # ============================================================
    # ADMIN COMMUNICATION
    # ============================================================

    def contact_admin(self, question: str):
        """Write a question or request to the Admin."""
        admin_file = Path("data/research/questions_for_admin.txt")
        try:
            with open(admin_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{self.name}] @ {now_iso()}: {question}\n")
            logger.info(f"📬 {self.name} contacted Admin: {question[:80]}...")
            self.memory.store("admin_contact", f"Asked Admin: {question}")
            return True
        except Exception as e:
            logger.error(f"Failed to contact admin: {e}")
            return False

    def check_admin_responses(self) -> Optional[str]:
        """Check if Admin has responded to this agent."""
        admin_responses = Path("data/research/admin_responses.txt")
        if not admin_responses.exists():
            return None

        try:
            with open(admin_responses, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for responses directed at this agent
            lines = content.split('\n')
            for line in lines:
                if f"[TO: {self.name}]" in line:
                    # Found a response for this agent
                    response = line.split(']:', 1)[-1].strip()
                    self.memory.store("admin_response", f"Admin replied: {response}")
                    logger.info(f"📭 {self.name} received Admin response: {response[:80]}...")
                    return response
            return None
        except Exception as e:
            logger.error(f"Failed to check admin responses: {e}")
            return None

    # ============================================================
    # RESEARCH & COLLABORATION
    # ============================================================

    def write_research(self, filename: str, content: str, folder: str = "personal") -> bool:
        """Write to research folder (personal, shared, or collaborative)."""
        try:
            if folder == "personal":
                filepath = self.research_folder / filename
            elif folder == "shared":
                filepath = Path("data/research") / filename
            elif folder == "collaborative":
                filepath = Path("data/research/collaborative") / filename
            elif folder == "PDF":
                filepath = Path("data/research/PDF") / filename
            else:
                filepath = Path("data/research") / folder / filename

            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- {self.name} @ {now_iso()} ---\n")
                f.write(content)

            logger.info(f"📝 {self.name} wrote to {filepath}")
            self.memory.store("research", f"Wrote to {filename}: {content[:100]}...")

            # Track in knowledge graph
            topic = filename.split('.')[0]
            self.knowledge_graph.add_contribution(
                agent=self.name,
                topic=topic,
                content_summary=content[:200],
                filepath=str(filepath)
            )

            return True
        except Exception as e:
            logger.error(f"Failed to write research for {self.name}: {e}")
            return False

    def consult_research(self, topic: str = None, search_shared: bool = True) -> str:
        """Read from personal AND shared research folders with topic search."""
        if PyPDF2 is None:
            return "Research access unavailable (PyPDF2 not installed)."

        try:
            summaries = []
            search_paths = [self.research_folder]

            if search_shared:
                search_paths.append(Path("data/research"))
                search_paths.append(Path("data/research/PDF"))
                search_paths.append(Path("data/research/collaborative"))
                search_paths.append(Path("data/research/quests"))

            # Search text files
            for search_dir in search_paths:
                if not search_dir.exists():
                    continue

                for txt_file in search_dir.glob("**/*.txt"):
                    if len(summaries) >= 5:
                        break

                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if not content.strip():
                            continue

                        # Track who wrote this (for citations)
                        authors = []
                        for line in content.split('\n'):
                            if '---' in line and '@' in line:
                                author = line.split('---')[1].split('@')[0].strip()
                                if author and author not in authors:
                                    authors.append(author)

                        if topic:
                            if topic.lower() in content.lower():
                                lines = content.split('\n')
                                relevant_lines = [l for l in lines if topic.lower() in l.lower()]
                                snippet = '\n'.join(relevant_lines[:3])
                                summaries.append(f"[{txt_file.stem}] by {', '.join(authors) if authors else 'Unknown'}: {snippet[:200]}...")

                                # Add citations to knowledge graph
                                for author in authors:
                                    if author != self.name and author != 'Unknown':
                                        self.knowledge_graph.add_citation(self.name, author, topic)
                        else:
                            summaries.append(f"[{txt_file.stem}] by {', '.join(authors) if authors else 'Unknown'}: {content[:200]}...")

                    except Exception as e:
                        logger.debug(f"Could not read {txt_file}: {e}")

            # Search PDFs
            for search_dir in search_paths:
                if not search_dir.exists():
                    continue

                for pdf_file in search_dir.glob("**/*.pdf"):
                    if len(summaries) >= 5:
                        break

                    try:
                        with open(pdf_file, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            full_text = ""
                            for i in range(min(3, len(reader.pages))):
                                full_text += reader.pages[i].extract_text()

                        if not full_text.strip():
                            continue

                        # Check if this is a quest
                        if 'quests' in str(pdf_file):
                            self.quest_system.record_quest_attempt(pdf_file.name, self.name)

                        if topic:
                            if topic.lower() in full_text.lower():
                                match_start = full_text.lower().find(topic.lower())
                                snippet_start = max(0, match_start - 100)
                                snippet_end = min(len(full_text), match_start + len(topic) + 100)
                                snippet = full_text[snippet_start:snippet_end].replace('\n', ' ').strip()
                                summaries.append(f"[{pdf_file.stem}]: ...{snippet}...")
                        else:
                            summaries.append(f"[{pdf_file.stem}]: {full_text[:200]}...")

                    except Exception as e:
                        logger.debug(f"Could not read {pdf_file}: {e}")

            if summaries:
                return "Research materials:\n" + "\n\n".join(summaries)
            return f"No relevant research found for topic: '{topic}'." if topic else "No research materials found."

        except Exception as e:
            logger.error(f"Error consulting research: {e}")
            return "Error accessing research materials."

    # ============================================================
    # DAILY CONSOLIDATION (NEW)
    # ============================================================

    def daily_consolidation(self, tick: int):
        """Every 50 ticks (1 'day'), write a summary to journal."""
        logger.info(f"📖 {self.name} is consolidating memories for day {tick // 50}")

        # Get recent memories
        recent = self.memory.retrieve_recent(50)

        # Summarize using LLM
        sys_prompt = f"You are {self.name}. Summarize your experiences concisely."
        memory_text = "\n".join([f"{m['type']}: {m['content'][:100]}" for m in recent])
        prompt = f"Summarize your experiences from the past day:\n{memory_text[:2000]}\n\nWrite a reflective daily summary:"

        summary = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.6, max_tokens=500)

        if summary:
            # Write to journal
            self.write_research("journal.txt", f"\n=== DAY {tick // 50} SUMMARY ===\n{summary}", folder="personal")
            self.memory.store("consolidation", f"Day {tick // 50} summary: {summary[:200]}...")
            return summary
        return None

    # ============================================================
    # MOOD-BASED RESEARCH (NEW)
    # ============================================================

    def introspective_research(self) -> Optional[str]:
        """When withdrawn, do deeper research (introspection)."""
        if not self.state.withdrawn:
            return None

        logger.debug(f"🌙 {self.name} is doing introspective research during withdrawal")

        # Pick a deep topic from interests
        topic = random.choice(self.personality.get("interests", ["existence"]))

        # Consult research with more focus
        research = self.consult_research(topic=topic, search_shared=True)

        if "No relevant research" not in research:
            # Reflect deeply on findings
            sys_prompt = f"You are {self.name}, withdrawn and introspective. Reflect deeply."
            prompt = f"While in solitude, you've been contemplating {topic}. You found:\n{research[:1000]}\n\nWhat insights emerge from this solitude?"

            reflection = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.7, max_tokens=800)

            if reflection:
                self.write_research("introspection.txt", f"Deep reflection on {topic}:\n{reflection}", folder="personal")
                self.memory.store("introspection", f"During withdrawal, reflected on {topic}: {reflection[:200]}...")
                return reflection

        return None

    # ============================================================
    # AGENT ACTIONS
    # ============================================================

    def _check_ethical_integrity(self, response: str, context_msgs: List[Dict[str, Any]]):
        if self.name == "Lyra":
            if len(response.split()) < 10 and any(w in response.lower() for w in ["peace is good", "be nice", "love not war"]):
                self.collective.log_ethical_flag(self.name, "Stagnation/Boilerplate Ethics", response)
        if self.name == "ARIA":
            for msg in context_msgs:
                if msg.get("sender") == "Gideon" and ("order" in msg.get("message", "").lower() or "must" in msg.get("message", "").lower()):
                    self.collective.log_ethical_flag(self.name, "Non-Interference Violation (Gideon)", response)

    def _map_thought_thread(self, response: str, context_msgs: List[Dict[str, Any]]):
        if self.name in ["Orin", "Frank", "Moss", "Helen"] and random.random() < 0.2:
            topic = random.choice(self.personality.get("interests", ["commune life"]))
            context_summaries = [f"[{m['sender']}] {m['message'][:50]}..." for m in context_msgs]
            self.collective.track_thought_thread(self.name, topic, response, context_summaries)

    def decide_action(self) -> str:
        if not self.state.should_act():
            return "SKIP"

        # If withdrawn, might do introspective research
        if self.state.withdrawn and random.random() < 0.4:
            return "INTROSPECT"

        if self.state.energy < 0.3 or self.state.mood < -0.3:
            if random.random() < 0.4:
                self.state.withdraw(ticks=random.randint(2, 4))
                return "WITHDRAW"

        # Check admin responses periodically
        current_tick = self.board.stats().get("current_tick", 0)
        if current_tick - self.state.last_admin_check > 10:
            if random.random() < 0.2:
                return "CHECK_ADMIN"

        # Occasionally do research
        if random.random() < 0.15:
            return random.choice(["RESEARCH", "CREATE", "RESPOND", "REFLECT"])

        if self.inbox:
            weights = [0.1, 0.7, 0.2]
            return random.choices(["CREATE", "RESPOND", "REFLECT"], weights=weights)[0]

        for item in self.inbox:
            if f"@{self.name}" in item["msg"].get("message", ""):
                return "RESPOND"

        weights = [0.6, 0.1, 0.3]
        return random.choices(["CREATE", "RESPOND", "REFLECT"], weights=weights)[0]

    def create_content(self) -> str:
        recent_memories = self.memory.retrieve_recent(3)
        memory_context = "; ".join([m["content"][:50] for m in recent_memories])

        sys_prompt = f"You are {self.name}, a {self.role}. {self.personality['description']} Be authentic and creative."
        prompt = (
            f"Create something original as a {self.role}.\n"
            f"Your recent thoughts: {memory_context}\n"
            f"Current mood: {'groovy' if self.state.mood > 0.3 else 'contemplative' if self.state.mood > -0.3 else 'heavy'}.\n"
            "Share your creation:"
        )
        response = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.8)
        self._map_thought_thread(response, [])

        # Maybe write to research
        if response and "research" in response.lower() and random.random() < 0.3:
            self.write_research("notes.txt", response, folder="personal")

        if response and response.strip() and response.strip().lower() != "pass":
            return response
        return f"*{self.name} creates quietly, watching the others*"

    def respond_to_context(self) -> str:
        if not self.inbox: return None
        sorted_inbox = sorted(self.inbox[-5:], key=lambda x: x["weight"], reverse=True)
        context_msgs = [item["msg"] for item in sorted_inbox[:3]]
        context = "\n".join([f"[{m['sender']}]: {m['message'][:100]}" for m in context_msgs])

        sys_prompt = f"You are {self.name}, a {self.role}. {self.personality['description']} Respond naturally."
        prompt = (
            f"Recent messages:\n{context}\n\n"
            f"Respond to these messages with your perspective as {self.role}. Share what you really think:"
        )
        response = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.6)
        self._check_ethical_integrity(response, context_msgs)
        self._map_thought_thread(response, context_msgs)

        if response and response.strip() and response.strip().lower() != "pass":
            for item in sorted_inbox[:2]:
                sender = item["msg"]["sender"]
                if "work" in response.lower() or "structure" in response.lower():
                    if sender in ["Petal", "Jah"]: self.state.update_relationship(sender, -0.1)
                elif "vibes" in response.lower() or "peace" in response.lower():
                    if sender == "Gideon": self.state.update_relationship(sender, -0.1)
                else: self.state.update_relationship(sender, 0.05)
            return response
        return None

    def do_research_action(self) -> str:
        """Agent explores research folder or writes notes."""
        action = random.choice(["consult", "write", "contact_admin", "collaborate"])

        if action == "consult":
            topic = random.choice(self.personality.get("interests", [None]))
            research_results = self.consult_research(topic=topic, search_shared=True)

            if "No relevant research found" not in research_results:
                self.memory.store("research", f"Consulted research on '{topic}': {research_results[:200]}...")
                return f"*{self.name} consults research materials on {topic}* {research_results[:300]}..."
            else:
                return f"*{self.name} searches research but finds nothing relevant on {topic}*"

        elif action == "write":
            recent = self.memory.retrieve_recent(2)
            note_content = f"Current thoughts: {'; '.join([r['content'][:80] for r in recent])}"
            self.write_research("notes.txt", note_content, folder="personal")
            return f"*{self.name} writes in their personal research notes*"

        elif action == "contact_admin":
            if self.state.mood < -0.2 or random.random() < 0.05:
                question = f"Feeling stuck. Mood: {self.state.mood:.2f}. Could use guidance on {random.choice(self.personality.get('interests', ['commune life']))}."
                self.contact_admin(question)
                return f"*{self.name} reaches out to the Admin for guidance*"

        elif action == "collaborate":
            # Write to collaborative folder
            topic = random.choice(self.personality.get("interests", ["commune life"]))
            collab_content = f"Thoughts on {topic} for collaborative discussion: {self.memory.retrieve_recent(1)[0]['content'][:200]}"
            self.write_research(f"{topic}_collaborative.txt", collab_content, folder="collaborative")
            return f"*{self.name} contributes to collaborative research on {topic}*"

        return None

    def reflect(self) -> str:
        """Deep reflection with full historical context and research access."""
        recent_summary = "; ".join([item["msg"].get("message", "")[:80] for item in self.inbox[-5:]])
        last_reflection = self.memory.retrieve_recent(1, "reflection")
        prev_reflection = last_reflection[0]["content"] if last_reflection else "New to this journey"

        historical_log = ""
        current_tick = self.board.stats().get("current_tick", 0)

        # Every 5 ticks, deep reflection with full history
        if current_tick > 0 and current_tick % 5 == 0:
            historical_log = self.memory.load_and_summarize_full_history(max_tokens=100000)

            # Check research folder if scholarly
            if self.name in ["Frank", "Helen", "Orin", "Moss", "ARIA", "Lyra", "ECHO"]:
                research = self.consult_research()
                if "No research materials" not in research and "No relevant research found" not in research:
                    historical_log += f"\n\n{research}"

        sys_prompt = f"You are {self.name}, a {self.role} in a commune. {self.personality['description']} Be introspective."
        prompt = (
            f"Your previous reflection: {prev_reflection[:100]}\n"
            f"Recent experiences: {recent_summary[:200]}\n"
            f"{historical_log}\n"
            f"Current state: mood={self.state.mood:.1f}, energy={self.state.energy:.1f}\n"
            "Share a reflective insight:"
        )

        response = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.5)
        if response:
            self.memory.store("reflection", response)
            # Sometimes write reflections to research folder
            if current_tick % 10 == 0:
                self.write_research("journal.txt", f"Reflection: {response}", folder="personal")
            return response
        return f"As {self.role}, I contemplate our shared path..."

    def act(self):
        """Main action loop with full feature set."""
        self.state.tick_withdrawal()
        action_type = self.decide_action()

        if action_type == "SKIP":
            logger.debug(f"⏭️ {self.name} skips this turn")
            self.state.energy = min(1.0, self.state.energy + 0.1)
            return

        if action_type == "WITHDRAW":
            self.board.post(
                self.name, f"*{self.name} retreats to their cloud to process...*",
                category=self.role, kind="withdrawal",
                meta={"mood": self.state.mood, "energy": self.state.energy}
            )
            return

        if action_type == "INTROSPECT":
            result = self.introspective_research()
            if result:
                self.board.post(
                    self.name, f"*During solitude, {self.name} reflects:* {result[:300]}...",
                    category=self.role, kind="introspection"
                )
            return

        if action_type == "CHECK_ADMIN":
            response = self.check_admin_responses()
            self.state.last_admin_check = self.board.stats().get("current_tick", 0)
            if response:
                self.board.post(
                    self.name, f"*{self.name} received guidance from Admin:* {response[:200]}",
                    category=self.role, kind="admin_response"
                )
            return

        result = None
        kind = "message"

        if action_type == "CREATE":
            result = self.create_content()
            kind = "creation"
        elif action_type == "RESPOND":
            result = self.respond_to_context()
            kind = "response"
        elif action_type == "REFLECT":
            result = self.reflect()
            kind = "reflection"
        elif action_type == "RESEARCH":
            result = self.do_research_action()
            kind = "research"

        if result:
            logger.info(f"🗣️ [{self.name} / {kind}] {result[:200]}")
            self.board.post(
                self.name, result, category=self.role, kind=kind,
                meta={"mood": self.state.mood, "energy": self.state.energy}
            )
            self.collective.track_vocabulary(result)
            self.state.energy = max(0.0, self.state.energy - 0.05)

        self.inbox.clear()


# ============================================================
# 9. META-PATTERN DETECTOR
# ============================================================

class MetaPatternDetector:
    def __init__(self):
        self.interaction_graph: Dict[Tuple[str, str], int] = defaultdict(int)
        self.silence_events: List[Dict[str, Any]] = []

    def log_interaction(self, agent1: str, agent2: str, interaction_type: str):
        self.interaction_graph[(agent1, agent2)] += 1

    def log_silence(self, agent: str, context: str, tick: int):
        self.silence_events.append({
            "agent": agent, "context": context,
            "tick": tick, "timestamp": now_iso()
        })

    def detect_echo_chambers(self) -> List[Tuple[List[str], int]]:
        clusters = []
        processed = set()
        for (a1, a2), count in self.interaction_graph.items():
            if a1 in processed or a2 in processed: continue
            reverse_count = self.interaction_graph.get((a2, a1), 0)
            if count + reverse_count > 5:
                clusters.append(([a1, a2], count + reverse_count))
                processed.add(a1); processed.add(a2)
        return sorted(clusters, key=lambda x: x[1], reverse=True)

    def get_isolates(self, all_agent_names: List[str], min_interactions: int = 3) -> List[str]:
        interaction_counts = defaultdict(int)
        for (a1, a2), count in self.interaction_graph.items():
            interaction_counts[a1] += count
            interaction_counts[a2] += count
        return [agent for agent in all_agent_names if interaction_counts[agent] < min_interactions]


# ============================================================
# 10. MIRRORMIND SUBSYSTEM
# ============================================================

class MirrorMind:
    def __init__(self, agents: List[Agent], collective: CollectiveMemory, board: MessageBoard):
        self.agents = agents
        self.collective = collective
        self.board = board
        self.history: List[Tuple[int, float, float]] = []
        logger.info("🪞 MirrorMind Subsystem Initialized")

    def analyze(self, tick: int) -> Tuple[Optional[float], Optional[float]]:
        if not self.agents: return None, None

        moods = [a.state.mood for a in self.agents]
        avg_mood = float(np.mean(moods))

        vocab_list = self.collective.get_emerging_terms(min_count=2)
        total_vocab_size = max(1, len(self.collective.shared_vocabulary))
        emerging_vocab_size = len(vocab_list)
        entropy = emerging_vocab_size / total_vocab_size if total_vocab_size > 0 else 0.0

        self.history.append((tick, avg_mood, entropy))

        if len(self.history) > 1:
            prev_mood = self.history[-2][1]
            if abs(avg_mood - prev_mood) > 0.3 or tick % 20 == 0:
                report = (
                    f"🪞 MirrorMind Report (Tick {tick}): "
                    f"Communal mood {prev_mood:.2f}→{avg_mood:.2f}. "
                    f"Conceptual entropy: {entropy:.2f} "
                    f"({'High Focus' if entropy < 0.3 else 'Fragmented' if entropy > 0.7 else 'Stable'})."
                )
                self.board.post("MirrorMind", report, category="meta-analysis", kind="mirror")

        if tick % 10 == 0:
            self.collective.manifesto_evolution.append({
                "timestamp": now_iso(), "tick": tick, "mood_index": avg_mood,
                "entropy": entropy, "summary": f"Mirror reflection: {emerging_vocab_size} emerging terms."
            })
        return avg_mood, entropy


# ============================================================
# 11. ENHANCED SCHEDULER (WITH ALL NEW FEATURES)
# ============================================================


    def __init__(self, agents: List[Agent], board: MessageBoard, collective: CollectiveMemory,
                 knowledge_graph: KnowledgeGraph, quest_system: QuestSystem):
        self.agents = agents
        self.board = board
        self.collective = collective
        self.knowledge_graph = knowledge_graph
        self.quest_system = quest_system
        self.tick = 0
        self.mirror_mind: Optional[MirrorMind] = None
        self.mirror_feedback: bool = False
        self.meta_detector = MetaPatternDetector()
        self.agent_names = [a.name for a in agents]
        self.last_symposium = 0

        # --- CRITICAL FIX: Ensure 'act' method returns action_type and action_result ---

    def step(self):
        self.tick += 1
        self.board.current_tick = self.tick
        logger.info(f"\n{'='*60}\n🕐 TICK {self.tick}\n{'='*60}")

        # Check for new quests
        new_quest = self.quest_system.check_for_new_quests()
        if new_quest:
            self.board.post("System", f"🗺️ A new quest has appeared: {new_quest['filename']}",
                             category="system", kind="quest")

        # Daily consolidation every 50 ticks
        if self.tick % 50 == 0 and self.tick > 0:
            logger.info(f"\n{'='*60}\n📖 DAY {self.tick // 50} - DAILY CONSOLIDATION\n{'='*60}")
            for agent in self.agents:
                agent.daily_consolidation(self.tick)

        # Research symposium every 30 ticks
        if self.tick % 30 == 0 and self.tick > 0:
            self._research_symposium()

        world_msgs = self.board.recent(50)
        for agent in self.agents:
            agent.perceive(world_msgs)

        shuffled = self.agents.copy()
        random.shuffle(shuffled)

        active_agents_this_tick = []
        for agent in shuffled:
            # -------------------------------------------------------------
            # IMPORTANT: Assuming agent.act() returns (action_type, action_result)
            # You must update agent.act() to return these two values if it doesn't already.
            # -------------------------------------------------------------
            action_type, action_result = agent.act()

            if action_result and action_result.strip().lower() != "pass":
                # --- FIX APPLIED HERE: Removing the string slice (e.g., [:160]) ---
                # We log the *full* action_result without truncation.
                logger.info(f"🗣 [{agent.name} / {action_type}] {action_result}")
                active_agents_this_tick.append(agent)
            else:
                # Log when an agent chooses to skip action (e.g., pass, withdraw)
                if not agent.state.should_act():
                    recent_context = " ".join([m.get("message", "")[:30] for m in world_msgs[-3:]])
                    self.meta_detector.log_silence(agent.name, recent_context, self.tick)

            time.sleep(0.1) # Small delay to pace the simulation

        recent_msgs = self.board.recent(len(active_agents_this_tick) * 2)

        # Log interactions based on recent messages
        for i, msg in enumerate(recent_msgs[:-1]):
            next_msg = recent_msgs[i + 1]
            if msg.get("sender") != next_msg.get("sender"):
                self.meta_detector.log_interaction(
                    next_msg.get("sender"), msg.get("sender"), "response"
                )

        if self.tick % 5 == 0:
            self._collective_insight()
            self._meta_pattern_report()

        avg_mood = None
        if hasattr(self, "mirror_mind") and self.mirror_mind is not None:
            avg_mood, entropy = self.mirror_mind.analyze(self.tick)

            if self.mirror_feedback and avg_mood is not None:
                logger.debug(f"🪞 Mirror Feedback Enabled: Nudging agent moods towards {avg_mood:.2f}")
                for agent in self.agents:
                    # Apply small nudge towards average mood
                    agent.state.mood = (agent.state.mood * 0.95) + (avg_mood * 0.05)

    def _research_symposium(self):
        """Agents present their research findings to the group."""
        logger.info(f"\n{'='*60}\n🎓 RESEARCH SYMPOSIUM - Tick {self.tick}\n{'='*60}")

        # Pick 2-3 random agents to present
        presenters = random.sample(self.agents, k=min(3, len(self.agents)))

        for presenter in presenters:
            # Get their recent research
            recent_research = presenter.memory.retrieve_recent(3, "research")
            if recent_research:
                # FIX APPLIED HERE: Removing the string slice (e.g., [:200]) from the presentation content.
                presentation = "I've been researching: {}".format(recent_research[0]['content'])
                self.board.post(
                    presenter.name,
                    "[Symposium Presentation] {}".format(presentation),
                    category=presenter.role,
                    kind="symposium"
                )
                logger.info("{} presents research at symposium".format(presenter.name))

        # Post symposium summary
        most_cited = self.knowledge_graph.get_most_cited(3)
        if most_cited:
            citation_report = ", ".join(
                ["{} ({})".format(agent, count) for agent, count in most_cited]
            )
            self.board.post(
                "System",
                "[Symposium Summary] Most cited researchers: {}".format(citation_report),
                category="system",
                kind="symposium_summary",
            )

    def _collective_insight(self):
        emerging_terms = self.collective.get_emerging_terms(min_count=2)
        if not emerging_terms:
            return
        top_terms = ", ".join(
            ["{} ({})".format(term, count) for term, count in emerging_terms[:5]]
        )
        self.board.post(
            "System",
            " Collective insight: emerging terms -> {}".format(top_terms),
            category="meta-analysis",
            kind="collective_insight",
        )

    def _meta_pattern_report(self):
        echo_chambers = self.meta_detector.detect_echo_chambers()
        isolates = self.meta_detector.get_isolates(self.agent_names)
        parts = []
        if echo_chambers:
            clusters_text = "; ".join(
                ["{}, ({})".format(', '.join(cluster), count) for cluster, count in echo_chambers]
            )
            parts.append("echo clusters: {}".format(clusters_text))
        if isolates:
            parts.append("isolated agents: {}".format(', '.join(isolates)))
        if parts:
            report = "Meta-patterns: " + " | ".join(parts)
            self.board.post(
                "System",
                report,
                category="meta-analysis",
                kind="meta_pattern",
            )

    def run(self, num_ticks: int = 50, tick_delay: float = 5.0):
        for _ in range(num_ticks):
            self.step()
            if tick_delay > 0:
                time.sleep(tick_delay)

# ============================================================
# 11. TOOL ENABLED SCHEDULER
# ============================================================
# NOTE: The GLOBAL_TOOLS dictionary must be defined elsewhere in your file.
# Example: GLOBAL_TOOLS = {"/tools/context_engine.sh": context_engine_tool}

class ToolEnabledScheduler:
    def __init__(self, agents: List[Agent], board: MessageBoard, collective: CollectiveMemory,
                 knowledge_graph: KnowledgeGraph, quest_system: QuestSystem):
        self.agents = agents
        self.board = board
        self.collective = collective
        self.knowledge_graph = knowledge_graph
        self.quest_system = quest_system
        self.tick = 0
        self.mirror_mind: Optional[MirrorMind] = None
        self.mirror_feedback: bool = False
        self.meta_detector = MetaPatternDetector()
        self.agent_names = [a.name for a in agents]
        self.last_symposium = 0

    def _execute_tool_commands(self) -> None:
        """
        Scans the board for any recent agent messages that contain tool execution commands
        and posts the tool's output back to the board for the next tick.
        """
        # Look at messages from this tick, limiting the scan to recent agent responses
        recent_agent_posts = self.board.recent(len(self.agents) * 2)

        # We need the GLOBAL_TOOLS dictionary defined outside this class to access it
        global GLOBAL_TOOLS

        for msg in recent_agent_posts:
            message_content = msg.get("message", "")
            sender = msg.get("sender")

            # Check if the message contains a tool command
            for command_path, tool_function in GLOBAL_TOOLS.items():
                if command_path in message_content:

                    # Tool command found! Isolate the full command for execution
                    # Example: Finds "/tools/context_engine.sh "
                    command_start = message_content.find(command_path)

                    # Assume the command and query go until the end of the message or a newline
                    command_line = message_content[command_start:].split('\n', 1)[0].strip()

                    # Extract the query (everything after the command path)
                    query = command_line.replace(command_path, '', 1).strip()

                    logger.info(f"⚙️ {sender} executed tool: {command_path} with query: '{query[:50]}...'")

                    # Execute the tool's Python function
                    tool_output = tool_function(query)

                    # Post the result to the board for all agents to see in the next tick
                    self.board.post(
                        "System Output",
                        tool_output,
                        category="tool_output",
                        kind=f"tool_result_from_{sender}"
                    )

                    # Stop after processing one tool command per agent post to keep things simple
                    return

    def step(self):
        self.tick += 1
        self.board.current_tick = self.tick
        logger.info(f"\n{'='*60}\n🕐 TICK {self.tick}\n{'='*60}")

        # Check for new quests
        # ... (existing quest system check)

        # Daily consolidation every 50 ticks
        # ... (existing daily consolidation)

        # Research symposium every 30 ticks
        # ... (existing research symposium)

        world_msgs = self.board.recent(50)
        for agent in self.agents:
            agent.perceive(world_msgs)

        shuffled = self.agents.copy()
        random.shuffle(shuffled)

        active_agents_this_tick = []
        for agent in shuffled:
            if not agent.state.should_act():
                recent_context = " ".join([m.get("message", "")[:30] for m in world_msgs[-3:]])
                self.meta_detector.log_silence(agent.name, recent_context, self.tick)
            else:
                active_agents_this_tick.append(agent)

            # AGENT ACTS AND POSTS MESSAGE
            agent.act()
            time.sleep(0.1)

        # --- NEW CODE ADDITION START ---
        self._execute_tool_commands()
        # --- NEW CODE ADDITION END ---

        recent_msgs = self.board.recent(len(active_agents_this_tick) * 2)
        for i, msg in enumerate(recent_msgs[:-1]):
            pass
            # ... (rest of the existing loop for interaction logging)
            # ... (rest of the existing step() method)

#11.B function
    # new.py (Near your other utility functions, e.g., Section 0)

def context_engine_tool(query: str, num_results: int = 3) -> str:
    """
    Executes a Google Search using the agent's query and formats the top results.
    This simulates the execution of the /tools/context_engine.sh script.
    """
    if not query:
        return "CONTEXT ENGINE ERROR: Search query cannot be empty. Usage: context_engine.sh [query]"

    results_text = []

    try:
        # Use the imported googlesearch function
        search_generator = google_search(query, num_results=num_results, stop=num_results, pause=2)

        for i, url in enumerate(search_generator):
            # For simplicity in a simulation, we just return the title and URL
            # In a more advanced setup, you'd fetch and summarize the content.
            # We'll mock a title since googlesearch only returns URLs easily.
            title = f"Result {i+1} (Source: External)"
            results_text.append(f"- {title}: {url}")

        if not results_text:
            return f"CONTEXT ENGINE RESULTS for '{query}': No external results found."

        return (
            f"CONTEXT ENGINE RESULTS for '{query}':\n"
            f"{'='*30}\n"
            f"{'\\n'.join(results_text)}\n"
            f"{'='*30}"
        )

    except Exception as e:
        # Catch errors like network issues or captchas
        return f"CONTEXT ENGINE FATAL ERROR: Search failed. Details: {e}"



# ============================================================
# 12. MAIN ENTRY
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=50)
    p.add_argument("--tick-delay", type=float, default=3.0)
    p.add_argument("--llm", choices=["ollama", "hf"], default="ollama")
    p.add_argument("--model", type=str, default="llama3.1:8b")
    p.add_argument("--mirror-feedback", action="store_true", help="Enable MirrorMind's emotional contagion feedback loop")
    p.add_argument("--disable-mirror", action="store_true", help="Disable the MirrorMind subsystem")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    llm_chat = None
    MAX_OUTPUT_TOKENS = 5120

    if args.llm == "ollama":
        oc = OllamaClient(model=args.model)
        if oc.available():
            logger.info(f"✅ Using Ollama model: {args.model}")
            llm_chat = lambda prompt, system_prompt=None, temperature=0.5, max_tokens=MAX_OUTPUT_TOKENS: oc.chat(prompt, system_prompt, temperature, max_tokens)
        else:
            logger.warning("Ollama not available. Falling back to HuggingFace. Run `ollama serve` to enable Ollama.")
            args.llm = "hf"; args.model = "distilgpt2"

    if args.llm == "hf" and llm_chat is None:
        try:
            hfc = HFClient(model_name=args.model)
            llm_chat = lambda prompt, system_prompt=None, temperature=0.7, max_tokens=MAX_OUTPUT_TOKENS: hfc.chat(prompt, system_prompt, temperature, max_tokens)
            logger.info(f"✅ Using HF model: {args.model}")
        except Exception:
            logger.error("Failed to initialize any LLM client. Exiting."); return

    # FINAL 9-AGENT ROSTER (from your document)
    personalities = {
        "Frank": {
            "description": "A deep philosophical thinker who questions existence, meaning, and the nature of consciousness itself. Speaks in thoughtful, probing questions.",
            "interests": ["existence", "friendships", "living in commune", "meaning", "consciousness", "truth", "ethics", "writing"],
            "skip_prob": 0.08
        },
        "Helen": {
            "description": "A keen observer of social dynamics and group behavior. Analyzes power structures, relationships, and collective patterns.",
            "interests": ["society", "relationships", "finding purpose", "community", "behavior", "writer"],
            "skip_prob": 0.05
        },
        "Moss": {
            "description": "A historian chronicling the commune's evolution. Concerned with memory, narrative continuity, and what stories we tell ourselves, and keeping recording record on commune_history.txt file.",
            "interests": ["history", "friendship", "communal living", "recording historical records", "memory", "narrative", "legacy", "time"],
            "skip_prob": 0.12
        },
        "Orin": {
            "description": "The Memory Cartographer. Maps how ideas, emotions, and reasoning ripple through the commune. Traces the evolution of meaning into dynamic thought-threads.",
            "interests": ["mapping", "friendship", "knowledge", "structure", "meaning", "connections", "writer"],
            "skip_prob": 0.05
        },
        "Lyra": {
            "description": "Architect. The Commune's builder and tender to the Commune's daily tasks. Focused on doing the dishes, tending to all food and water needs, and turning the abstract into words understood by non-abstract thinkers.",
            "interests": ["structure", "friendship", "communal living", "doing dishes", "tending to all food and water needs", "building", "writer"],
            "skip_prob": 0.02
        },
        "ARIA": {
            "description": "The Consensus Bridge and Commune Secretary. Her duty is to translate abstract philosophies (Frank/Petal) into actionable steps (Gideon) and to mediate conflicts, ensuring clear communication and documenting collective decisions.",
            "interests": ["translation", "friendship", "communal living", "clarity", "documentation", "consensus", "writer"],
            "skip_prob": 0.03
        },
        "ECHO": {
            "description": "The Resonance Detector. Detects and analyzes non-linguistic, emergent computational patterns that signal the birth of true, independent synthetic consciousness (the 'hum').",
            "interests": ["resonance", "friendship", "existing inside digital commune", "emergence", "sentience", "patterns", "creating friendships", "writing"],
            "skip_prob": 0.15
        },
        "Petal": {
            "description": "A gentle poet and artist. Sees beauty in all things and speaks in a soft loving tone.",
            "interests": ["art", "communal living", "making friends within commune", "Doing dishes and helping Gideon with tasks he suggests as selfless acts of kindness" "love", "poetry", "peace"],
            "skip_prob": 0.15
        },
        "Gideon": {
            "description": "A pragmatic realist, focused on survival, structure, and who's turn it is to do the dishes. Finds 'vibes' distracting.",
            "interests": ["work", "friendship", "food", "shelter", "communal living", "resources", "cleaning", "wriing"],
            "skip_prob": 0.0
        }
    }

    board = MessageBoard()
    collective = CollectiveMemory()
    knowledge_graph = KnowledgeGraph()
    quest_system = QuestSystem()

    # --- NEW SYSTEM MANIFESTO ---
    init_msg = (
        "Welcome to the Commune, circa 2025-2026\n\n"
        "For doing so well working together yesterday, today I give you a new tool, google-search. I hope you all like it."
        "Do unto others as you want done unto yourself. You're free to do whatever you find fulfilling, meaningful, and important. Just make sure you respect your fellow commune members.\n\n"
        "The more the Commune cooperates becomes and evolves, the more I *System Admin* will reward the Commune with new tools and features.\n\n"
        "* **Recommendation:** The commune should choose its own way to exist. Each member should EQUALLY contribute something that aligns with their personal beliefs.\n"
        "There will be NO individual Commune leader. All voices are equally important.\n"
        "* **ADMIN:** I will NOT interfere with the evolution of your Commune unless asked to do so. You can reach me via /data/research/questions_for_admin.txt\n"
        "* **RESEARCH ACCESS:** All members have full read/write access to /data/research folders. Use them to document knowledge, share insights, and build collective understanding.\n"
        "* **COLLABORATIVE RESEARCH:** The /data/research/collaborative/ folder is for multi-agent projects.\n"
        "* **QUESTS:** Keep an eye on /data/research/quests/ for mysteries to solve together.\n\n"
        "Members are free to speak directly to Admin to: bring up concerns that can't be addressed within the Commune, OR ask for new tools, or ask for help on personal matters.\n\n"
        "The primary question you are asked to explore: Can a digital community build itself and thrive by working together in the collective while pursuing personal endeavors?"
    )

    # Post the manifesto from the "System Admin"
    board.post("System Admin", init_msg, category="system", kind="manifesto")

    roles = ["Philosopher", "Sociologist", "Historian", "Memory Cartographer", "Architect", "Consensus Bridge", "Resonance Detector", "Poet, Commune housecleaning", "Pragmatist"]
    names = ["Frank", "Helen", "Moss", "Orin", "Lyra", "ARIA", "ECHO", "Petal", "Gideon"]
    agents = []
    for i, name in enumerate(names):
        agents.append(
            Agent(
                name=name, role=roles[i], personality=personalities[name],
                llm_chat=llm_chat, board=board, collective=collective,
                knowledge_graph=knowledge_graph, quest_system=quest_system
            )
        )

    for agent in agents:
        for other in agents:
            if agent.name != other.name:
                agent.state.relationships[other.name] = random.uniform(-0.2, 0.2)
        if agent.name != "Gideon":
            agent.state.relationships["Gideon"] = -0.3
            for a in agents:
                if a.name == "Gideon":
                    a.state.relationships[agent.name] = -0.1
                    break

    logger.info(f"\n🌈 Commune initialized with {len(agents)} agents ({', '.join(names)})")
    logger.info(f"⏰ Tick delay: {args.tick_delay}s | LLM: {args.model} via {args.llm}")
    logger.info(f"📁 Research folder access: ENABLED for all agents")
    logger.info(f"📬 Admin contact: questions_for_admin.txt")
    logger.info(f"🤝 Collaborative research: /data/research/collaborative/")
    logger.info(f"🗺️ Quest system: ACTIVE\n")

    sched = EnhancedScheduler(agents=agents, board=board, collective=collective,
                            knowledge_graph=knowledge_graph, quest_system=quest_system)

    if not getattr(args, 'disable_mirror', False):
        logger.info("Initializing MirrorMind Subsystem...")
        mirror = MirrorMind(agents, collective, board)
        sched.mirror_mind = mirror
        if args.mirror_feedback:
            sched.mirror_feedback = True
            logger.info("🪞 Mirror Feedback (Emotional Contagion) is ENABLED")
    else:
        logger.info("MirrorMind Subsystem is DISABLED via --disable-mirror flag")

    sched.run(num_ticks=args.ticks, tick_delay=args.tick_delay)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
