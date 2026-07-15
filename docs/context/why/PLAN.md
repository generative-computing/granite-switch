# Granite Switch — Business Goals (WHY)

> **CollaborativeDream layer: WHY** — the stable "why we build this" layer.
> Loaded into an agent's context **only on demand** via the `/cd-why` skill — never
> at startup. Keep it durable: this changes rarely (goals, not tasks).
>
> **Role in the spec.** This is the **WHY** layer: the problem we solve and the
> high-level needs of our stakeholders. It is deliberately free of
> implementation detail. It is the *hard* layer — these goals change slowly and
> everything downstream must justify itself against them. The **WHAT** layer
> (the versioned requirements under `docs/context/what/`, registry at
> [`INDEX.md`](../what/INDEX.md)) turns these goals into testable, versioned
> requirements; each requirement should trace back to a goal here.
>
> **Status:** Draft · **Version:** 0.2.0 · **Last updated:** 2026-07-14
> **Audience:** stakeholders, product, and engineering leads. Read this before
> the requirements.

---

## 1. The problem

Teams that want to ship a capable AI system today choose between two bad
options:

- **One big generalist model** — expensive to serve, and still mediocre at the
  specific tasks (retrieval-augmented answering, safety/guardian checks,
  hallucination detection, query rewriting) that a real application strings
  together.
- **Many small specialist models** — each accurate on its task, but
  operationally fragmented: separate deployments, separate memory, and a real
  cost every time the application moves from one capability to the next. So
  "just add another capability" is never free.

There is no *software-like* way to pick the capabilities you need, compose them
into a single artifact, deploy it once, and upgrade a single capability without
retraining or re-plumbing everything around it. That is the gap Granite Switch
exists to close.

## 2. Vision

**Build AI more like software.** Today a model's every capability is diffused
across one big block of weights — closer to a lump of clay than to a set of
LEGO bricks. Changing one behavior means retraining or re-prompting the whole
thing. Granite Switch brings the discipline and modularity of software
engineering to models: each capability is an **adapter function** — a small
piece trained to be an expert at one task, with a defined job to do. You pick
the capabilities you need, compose them into a single model you can deploy, and
later swap or upgrade any one of them on its own — a bit like the way software
dependencies are managed today. The result is AI that is easier to adapt,
cheaper to operate, and more predictable in production.

Granite Switch is one part of a coordinated stack (Granite base models · Granite
adapter-function libraries · [Mellea](https://mellea.ai) for turning capabilities
into typed, predictable functions).

## 3. Stakeholders and their needs

| Stakeholder | What they need from Granite Switch |
|---|---|
| **Application / model builders** | Assemble targeted capabilities without training; ship one artifact that runs in prototyping and production without a conversion step. |
| **Serving / infrastructure owners** | Many capabilities at the cost of one — fast, efficient, and economical to operate. |
| **Adapter-function authors (IBM Research & partners)** | Develop and benchmark a capability *independently*, without joint training against every other one, and publish it for others to compose. |
| **End users of the built applications** | Fast, accurate, and safe responses. |
| **Backend & OSS communities** | Standard-format contributions that load through the community's normal path, not out-of-tree forks. |
| **IBM (sponsor/business)** | Differentiate Granite; drive adoption of the Granite model + libraries + Mellea stack; grow an open developer community. |

## 4. Business goals

Each goal has a stable ID and a priority. **HARD** goals are non-negotiable —
they define what Granite Switch *is*, and a change to one is a change to the
product. **Directional** goals set the trajectory and are expected to be met
over time. The requirements under `docs/context/what/` trace up to these goals
(the mapping lives there, not here).

### BG-01 — Capabilities compose like software *(HARD)*

Independently built capabilities snap together into a single model you deploy
as one artifact, and any one of them can be swapped or upgraded on its own without
retraining the rest. Building AI should feel like assembling and maintaining software,
not re-casting a monolith each time a need changes.

### BG-02 — Build your own capabilities *(Directional)*

Users should be able to easily create and integrate their own capabilities alongside
built-in ones, extending models to address their unique needs. This extensibility should
be a first-class design principle, not an afterthought.

### BG-03 — Independent capability development *(HARD)*

Capabilities should be independently developed, trained, evaluated, and released.
Teams should be able to own individual capabilities with well-defined interfaces,
enabling parallel development, independent versioning, and continuous improvement
without coordinating retraining of the full model.

### BG-04 — Many capabilities at the cost of one *(HARD)*

Running a model with many composed capabilities should be about as fast and as
cheap as running a single model, with no penalty for moving between capabilities
mid-task. Efficiency is the reason to choose this over operating many separate
models.

### BG-05 — Small + targeted beats big + generalist *(Directional)*

A small model with the right specialized capabilities should be able to match or
surpass much larger generalist models on the tasks that matter, at a fraction of
the operating cost. Demonstrating this advantage is a core part of the value we
aim to prove.

### BG-06 — Deploy where the community already runs *(Directional)*

The composed model should load and run through popular inference backends using
their standard, familiar paths, so that adopting it feels like everyday software
rather than a special case. Its design should stay flexible enough to support a
diverse and evolving range of model architectures rather than a single fixed one.

### BG-07 — Open ecosystem *(HARD constraint)*

Granite Switch should foster an open ecosystem for composable AI. The project should
remain open source, reproducible, and compatible with evolving open-model ecosystems,
enabling external contributors to develop, share, and compose reusable capabilities.

---

*This document is the WHY layer and the source of the requirements under
`docs/context/what/` (registry: [`INDEX.md`](../what/INDEX.md)). When a
requirement can't be traced to a goal here, either the requirement or this
document is wrong — reconcile before building. To turn a need here into an
implementable requirement, add a `REQ-*.md` under
`docs/context/what/requirements/` and link it back to the goal above.*
