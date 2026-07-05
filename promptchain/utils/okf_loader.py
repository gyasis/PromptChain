"""OKFLoader — import & understand OKF knowledge bundles inside PromptChain (issue #42).

OKF (Open Knowledge Format) is a *format*: a directory of markdown concept files, each with YAML
frontmatter (a required `type`), reserved `index.md` (directory listing / progressive disclosure) and
`log.md` (history), and bundle-relative markdown links between concepts. Because it's a known format,
PromptChain can UNDERSTAND the bundle — its tree, the concept types, the cross-links, the curated index —
and inject it into a chain at the right granularity, mirroring how PrePrompt imports named prompts.

Two injection modes (see the helpers):
  • AGENTIC (folder target): give an AgenticStepProcessor the OUTLINE (what concepts exist: id/type/
    title/description) as always-on context + an `okf_read(id)` tool, so the agent NAVIGATES the bundle
    and pulls concepts on demand — OKF's progressive-disclosure purpose.
  • SEQUENTIAL (specific file): inject one concept's content into a targeted step.

stdlib only — a minimal frontmatter parser (no yaml dependency).
"""
import os
import re

_RESERVED = {"index.md", "log.md"}
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _parse_frontmatter(text):
    """Split '---\\nYAML\\n---\\nbody'. Minimal YAML: scalars + one-line [a, b] lists. Returns (fm, body)."""
    if text.startswith("---"):
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
        if m:
            fm = {}
            for line in m.group(1).splitlines():
                line = line.rstrip()
                if not line or line.lstrip().startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1); k = k.strip(); v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    v = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                else:
                    v = v.strip("'\"")
                fm[k] = v
            return fm, m.group(2).strip()
    return {}, text.strip()


class OKFConcept:
    def __init__(self, cid, frontmatter, body, path):
        self.id = cid                                  # Concept ID = bundle-relative path minus .md
        self.frontmatter = frontmatter
        self.body = body
        self.path = path
        self.type = frontmatter.get("type")
        self.title = frontmatter.get("title", cid)
        self.description = frontmatter.get("description", "")
        self.tags = frontmatter.get("tags", [])

    @property
    def links(self):
        """Bundle-relative concept IDs this concept links to (a link asserts a relationship)."""
        out = []
        for _, target in _LINK_RE.findall(self.body):
            t = target.split("#")[0].strip()
            if t.endswith(".md"):
                out.append(t.lstrip("/")[:-3])
        return out


class OKFLoader:
    """Load one or more OKF bundles and understand their structure. `bundle_dir` is the FOLDER TARGET."""

    def __init__(self, *bundle_dirs):
        self.dirs = [os.path.abspath(d) for d in bundle_dirs]
        self._concepts = {}     # cid -> OKFConcept
        self._index = {}        # dir -> index.md text
        self._scan()

    def _scan(self):
        for d in self.dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    if not fn.endswith(".md"):
                        continue
                    path = os.path.join(root, fn)
                    if fn == "index.md":
                        self._index[os.path.relpath(root, d)] = open(path, encoding="utf-8").read()
                        continue
                    if fn in _RESERVED:
                        continue
                    cid = os.path.relpath(path, d)[:-3].replace(os.sep, "/")
                    fm, body = _parse_frontmatter(open(path, encoding="utf-8").read())
                    self._concepts[cid] = OKFConcept(cid, fm, body, path)

    # -------- understanding / navigation --------
    def concepts(self):
        return sorted(self._concepts)

    def get(self, cid):
        return self._concepts.get(cid)

    def load(self, cid):
        """Concept body text (mirrors PrePrompt.load) — for a targeted sequential step."""
        c = self._concepts.get(cid)
        return c.body if c else ""

    def tree(self):
        """Nested dict of the bundle's concept tree (dir -> [concept ids])."""
        t = {}
        for cid in self.concepts():
            parts = cid.split("/")
            node = t
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node.setdefault("_concepts", []).append(cid)
        return t

    def outline(self):
        """The navigation MAP an agent reads first (progressive disclosure): each concept's id/type/
        title/description. This is what you give an agentic step so it knows what it can pull."""
        lines = []
        for cid in self.concepts():
            c = self._concepts[cid]
            desc = f" — {c.description}" if c.description else ""
            lines.append(f"- `{cid}` [{c.type or '?'}] {c.title}{desc}")
        return "\n".join(lines)

    def index_md(self, subdir="."):
        """The curated index.md (progressive disclosure), if the bundle authored one."""
        return self._index.get(subdir, "")

    # -------- rendering / injection --------
    def render(self, concept_ids=None, header=True):
        ids = list(concept_ids) if concept_ids else self.concepts()
        parts = []
        for cid in ids:
            c = self._concepts.get(cid)
            if not c:
                continue
            parts.append((f"## {c.title}\n" if header else "") + c.body)
        return "\n\n".join(parts)

    def context(self, concept_ids=None, title="Reference knowledge (OKF)"):
        body = self.render(concept_ids)
        return f"# {title}\n\n{body}" if body else ""


# ---------------------------------------------------------------- injection helpers
def okf_step(loader, concept_ids, task):
    """SEQUENTIAL step: inject the specified concept file(s), then the task. Returns an instruction string
    usable directly in PromptChain(instructions=[okf_step(...)])."""
    ctx = loader.context(concept_ids)
    return f"{ctx}\n\n---\nUsing the reference knowledge above, {task}" if ctx else task


def okf_agentic_context(loader, concept_ids=None, mode="outline"):
    """AGENTIC step (folder target): mode='outline' gives the navigation map (what exists) for progressive
    disclosure — pair with okf_reader_tool so the agent pulls concepts on demand; mode='full' inlines all
    concept bodies. Pass the result into an AgenticStepProcessor's instructions (always-on access)."""
    if mode == "full":
        return loader.context(concept_ids)
    return ("# Available knowledge (OKF) — call okf_read(concept_id) to read any of these:\n"
            + loader.outline())


def okf_reader_tool(loader):
    """Return an `okf_read(concept_id)` callable to register as a tool on an AgenticStepProcessor, so the
    agent reads concepts ON DEMAND (progressive disclosure) instead of ingesting the whole bundle."""
    def okf_read(concept_id: str) -> str:
        """Read one OKF concept's body by its Concept ID (see the outline)."""
        body = loader.load(concept_id)
        return body or f"(no concept '{concept_id}'; available: {', '.join(loader.concepts())})"
    return okf_read
