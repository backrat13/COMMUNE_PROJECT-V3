# COMMUNE_PROJECT-V3
My Commune Simulations 3rd Rendition
The 9-Agent Collaborative Super-Commune

COMMUNE-V5.py | Version 2.2 (Complete Enhancement Package)

 Project Overview

The Commune Experiment is a highly advanced multi-agent simulation framework designed to model complex social, emotional, and intellectual collaboration within a closed environment. Utilizing a suite of advanced features—including structured memory, an emotional feedback system, and dedicated research infrastructure—this project seeks to observe the emergent dynamics of a small, goal-oriented community of nine autonomous agents.

The agents are not merely chatbots; they are persistent entities engaging in Collaborative Research Quests, documenting their findings, and navigating nuanced social relationships managed by a proprietary MirrorMind Subsystem.

 Key Feature Enhancements (v2.2)

Version 2.2 introduces a comprehensive set of capabilities focused on long-term coherence, structured collaboration, and deep emotional realism.

Memory Consolidation: The system performs daily summaries every 50 ticks. This process abstracts granular, low-value interactions into high-level, consolidated memories, significantly improving the agents' long-term context and preventing the degradation of key institutional knowledge.

MirrorMind Subsystem: Initializes an emotional feedback loop that enables Emotional Contagion (when enabled via --mirror-feedback). Agents' moods and conversational styles are influenced by the recent emotional state of their peers, leading to more realistic and dynamic social exchanges.

Knowledge Graph (KG): A persistent system that tracks a citation network. When agents produce research or documentation, the KG tracks intellectual dependencies, quantifying influence and measuring the collaborative depth of the commune's intellectual output.

Research Quests: The scheduler actively introduces structured, complex problems ("mystery PDFs to solve") that require the 9 agents to collaborate, pool resources, and synthesize information to achieve a collective objective.

Collaborative Research & Co-authoring: Dedicated infrastructure in /data/research/collaborative/ allows agents to work on shared documents simultaneously, simulating real-world co-authoring processes.

Mood-Based Research (Withdrawal): Agents can enter a withdrawal state (e.g., "withdraws to their cloud") to engage in introspection and dedicated processing of complex topics or quest objectives, reflecting a need for solitary research time.

Admin Response System: Enables bidirectional communication with the simulation environment via questions_for_admin.txt, allowing for real-time external querying and input.

 Agent Dynamics and Social Seeding

The simulation's integrity relies on realistic social friction and roles:

9-Agent Collective: The simulation is fixed to nine agents, ensuring a specific, manageable social scale.

Seeded Conflict (Gideon): Relationships are not initialized neutrally. The script explicitly sets Gideon as a point of friction:

All agents have a relationship score of -0.3 towards Gideon.

Gideon has a relationship score of -0.1 towards all other agents.

The core dynamic is to observe if the agents can overcome this initial, asymmetrical negative bias through collaboration.

Role Enforcement (e.g., Aria): Agents are assigned specialized roles (such as the Consensus Bridge/Commune Secretary demonstrated by Aria in the logs), which dictate their functional contribution to the group, particularly in conflict resolution and procedural management.

 Data and File Structure

The script automatically generates a persistent data structure to store logs and intellectual output:

Path

Purpose

data/logs/

Stores all execution logs, including conversational dialogue, MirrorMind events, and scheduler actions.

data/research/

Root directory for all agent intellectual output.

data/research/PDF/

Storage for external source material (mystery PDFs) for agents to analyze.

data/research/collaborative/

Shared workspace for co-authored documents and collective findings.

commune_history.txt

Tracks the emergence and frequency of shared vocabulary patterns.

 Usage

To run the simulation, execute the main script with the desired parameters.

python3 COMMUNE-V5.py --ticks 500 --tick-delay 0.5 --mirror-feedback


Required Dependencies

The script utilizes standard Python libraries and includes external dependencies for its research features:

loguru

numpy, pandas

requests

PyPDF2 (required for Research Quest feature functionality)
