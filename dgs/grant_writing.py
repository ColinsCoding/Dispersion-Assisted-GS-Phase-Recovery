"""Grant writing guide and proposal outline generator -- tier 2 research.

WHAT "TIER 2" MEANS IN FEDERAL RESEARCH FUNDING:
  Tier 1 (exploratory / basic): NSF REU, NSF CAREER early-stage, seed grants
  Tier 2 (applied / translational): SBIR Phase I/II, NSF IIP, DARPA, ONR,
          NSF PFI (Partnerships for Innovation), DOE STTR
  Tier 3 (programmatic / system): DARPA programs, DOD contracts, large NSF centers

This module covers Tier 2 specifically:
  SBIR Phase I  ($275K / 6 months): prove feasibility, simulate the concept
  SBIR Phase II ($1.75M / 2 years): build the prototype, demo the system
  NSF PFI / IIP: university-industry partnership, commercialization path required

STRUCTURE OF A WINNING TIER-2 PROPOSAL (5 core sections):
  1. SIGNIFICANCE  -- why this problem matters (market, defense, science need)
  2. INNOVATION    -- what is NEW that no one else has done
  3. APPROACH      -- how you will do it (timeline, milestones, methods)
  4. TEAM          -- why YOUR team can execute (CVs, prior work)
  5. BUDGET        -- what money buys (personnel, equipment, travel, indirect)

The probability of winning:
  NSF SBIR Phase I accept rate: ~15%  (geometric: mean 1/0.15 = 6.7 submissions)
  DARPA BAA accept rate:        ~5-10%
  NSF CAREER:                   ~20%
  Rule: submit more, write sharper, build relationships before the call.

OOP: ProposalSection and ProposalOutline classes generate filled-in templates
for the specific case of this repo (TD-GS phase retrieval / RogueGuard receiver).
"""
import json
import os
import textwrap


# ── proposal section ─────────────────────────────────────────────────

class ProposalSection:
    """A single section of a grant proposal.

    Parameters
    ----------
    title : str    -- section name (e.g., "Significance")
    content : str  -- the body text
    word_limit : int -- typical NSF/SBIR word budget for this section
    """

    def __init__(self, title, content="", word_limit=500):
        self.title = title
        self.content = content
        self.word_limit = word_limit

    def word_count(self):
        return len(self.content.split())

    def is_within_limit(self):
        return self.word_count() <= self.word_limit

    def format(self, width=72):
        header = f"\n{'='*width}\n{self.title.upper()} ({self.word_count()}/{self.word_limit} words)\n{'='*width}"
        body = textwrap.fill(self.content, width=width) if self.content else "[TO BE WRITTEN]"
        return header + "\n" + body

    def __repr__(self):
        return f"ProposalSection('{self.title}', {self.word_count()} words)"


# ── full proposal outline ─────────────────────────────────────────────

class ProposalOutline:
    """A complete grant proposal outline with all five sections.

    Generates a filled-in template for the dispersion-assisted GS
    phase retrieval project (this repo), targeted at SBIR Phase I.

    Usage
    -----
    outline = ProposalOutline.sbir_phase1_gs_receiver()
    outline.print_all()
    outline.save("proposal_draft.txt")
    outline.save_json("proposal_draft.json")
    """

    def __init__(self, title, agency, mechanism, sections=None):
        self.title = title
        self.agency = agency
        self.mechanism = mechanism
        self.sections = sections or {}

    def add_section(self, key, section: ProposalSection):
        self.sections[key] = section

    def print_all(self):
        print(f"\n{'#'*72}")
        print(f"  PROPOSAL: {self.title}")
        print(f"  Agency: {self.agency}   Mechanism: {self.mechanism}")
        print(f"{'#'*72}")
        for section in self.sections.values():
            print(section.format())
        print(f"\n{'#'*72}")
        total = sum(s.word_count() for s in self.sections.values())
        print(f"  TOTAL WORD COUNT: {total}")
        print(f"{'#'*72}\n")

    def save(self, filepath):
        """Write plain-text draft to file."""
        lines = [f"PROPOSAL: {self.title}\n",
                 f"Agency: {self.agency}   Mechanism: {self.mechanism}\n"]
        for section in self.sections.values():
            lines.append(section.format())
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def save_json(self, filepath):
        """Save structured outline as JSON for later editing."""
        data = {
            "title": self.title, "agency": self.agency,
            "mechanism": self.mechanism,
            "sections": {k: {"title": s.title, "content": s.content,
                              "word_limit": s.word_limit}
                         for k, s in self.sections.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    @classmethod
    def load_json(cls, filepath):
        """Reload a saved outline from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        sections = {k: ProposalSection(v["title"], v["content"], v["word_limit"])
                    for k, v in data["sections"].items()}
        return cls(data["title"], data["agency"], data["mechanism"], sections)

    @classmethod
    def sbir_phase1_gs_receiver(cls):
        """Pre-filled SBIR Phase I outline for the TD-GS optical receiver."""
        outline = cls(
            title="Dispersion-Assisted Optical Phase Recovery for Carrier-Less Coherent Receivers",
            agency="NSF / DoD SBIR",
            mechanism="SBIR Phase I -- $275,000 / 6 months",
        )

        outline.add_section("significance", ProposalSection(
            title="Significance (Problem Statement)",
            word_limit=500,
            content=(
                "Coherent optical receivers offer superior noise performance and spectral "
                "efficiency for high-speed fiber communications, but they require a local "
                "oscillator laser synchronized to the carrier frequency. This increases "
                "system cost, complexity, and power consumption by 30-50% relative to "
                "direct-detection receivers. Carrier-less coherent detection -- recovering "
                "the complex optical field from intensity-only measurements -- would eliminate "
                "the local oscillator entirely. No commercially viable carrier-less coherent "
                "receiver currently exists. The proposed dispersion-assisted Gerchberg-Saxton "
                "(TD-GS) algorithm solves this problem by exploiting the time-stretch "
                "dispersive Fourier transform (TS-DFT) to encode spectral phase into "
                "temporal intensity, enabling phase recovery from two power measurements. "
                "This directly addresses the SBA priority area of photonic sensing and "
                "next-generation optical communications infrastructure."
            ),
        ))

        outline.add_section("innovation", ProposalSection(
            title="Innovation (What Is New)",
            word_limit=400,
            content=(
                "The innovation is the application of the Gerchberg-Saxton algorithm in the "
                "time domain (TD-GS), enabled by the TS-DFT dispersive fiber as a "
                "phase-diversity element. Prior art (Solli et al. 2009, Jalali group) "
                "demonstrated the concept in the optical domain for ultrashort pulse "
                "characterization. This proposal extends TD-GS to: (1) carrier-less "
                "coherent reception of BPSK and QAM-modulated signals at data rates "
                "above 10 Gbit/s; (2) a compact chirped fiber Bragg grating (CFBG) "
                "replacing km-scale fiber; (3) a real-time FPGA implementation of the "
                "GS iteration running at 1 GSample/s with sub-50-iteration convergence; "
                "and (4) a machine-learning front-end (FNO) that replaces the iterative "
                "algorithm for trained signal constellations. This combination has not "
                "been demonstrated in the prior literature."
            ),
        ))

        outline.add_section("approach", ProposalSection(
            title="Approach (Methods and Timeline)",
            word_limit=800,
            content=(
                "Month 1-2 (Simulation): Implement and validate the TD-GS algorithm in "
                "Python (numpy/scipy) and Google Colab. Simulate BPSK and 16-QAM signals "
                "at 1-10 GHz bandwidth. Characterize convergence vs. dispersion parameter D "
                "and signal-to-noise ratio. Milestone: Python simulator achieving BER < 1e-3 "
                "for BPSK at SNR > 15 dB. "
                "Month 3-4 (Bench Prototype): Assemble minimum viable optical bench: "
                "Thorlabs DET01CFC photodetector (1 GHz), 50:50 SMF-28 fiber coupler, "
                "CFBG providing D = -5000 ps^2, Keysight oscilloscope at 2.5 GSa/s. "
                "Capture I1(t) and I2(t) waveforms for CW and modulated laser inputs. "
                "Milestone: GS phase retrieval from real hardware data with error < 5 degrees RMS. "
                "Month 5-6 (FPGA Prototype and Report): Port the GS iteration to an "
                "FPGA (Xilinx Artix-7 or Zynq) using fixed-point arithmetic. Demonstrate "
                "real-time operation. Write Phase I final report and prepare Phase II proposal. "
                "Milestone: FPGA GS running at 100 MSample/s, 20 iterations, convergence verified."
            ),
        ))

        outline.add_section("team", ProposalSection(
            title="Team and Prior Work",
            word_limit=400,
            content=(
                "PI: [Your name], MS candidate, Electrical Engineering, UCLA. "
                "Research experience: EC ENGR 279AS (Jalali Lab special topics), "
                "phase retrieval simulation (GitHub: Dispersion-Assisted-GS-Phase-Recovery), "
                "35+ Python modules covering GS algorithm, Fourier optics, signal processing, "
                "and hardware BOM. "
                "Advisor: Prof. Bahram Jalali, UCLA Electrical Engineering. "
                "The Jalali Lab originated the TS-DFT concept (Solli et al. Nature 2007, "
                "Goda et al. Nature Photonics 2013) and holds key intellectual property "
                "in real-time optical measurement. "
                "Collaborators: Yiming Zhou, Callen MacPhee (Jalali Lab graduate researchers). "
                "Facilities: Jalali Lab optical bench (real-time oscilloscopes to 60 GHz, "
                "EDFA, tunable lasers, CFBGs), UCLA Electrical Engineering cleanroom."
            ),
        ))

        outline.add_section("budget", ProposalSection(
            title="Budget Justification ($275,000 Phase I)",
            word_limit=400,
            content=(
                "Personnel (60% of budget, $165,000): PI graduate research stipend "
                "12 months at $35,000; undergraduate research assistant 0.5 FTE at "
                "$18,000; indirect / fringe at estimated 30% of direct personnel costs. "
                "Equipment (25% of budget, $68,750): Chirped FBG CFBG module $800; "
                "Keysight oscilloscope 2.5 GSa/s $5,000; Xilinx Zynq FPGA development "
                "board $1,200; Thorlabs optical components and fiber $3,000; "
                "GPU workstation for ML baseline $8,000; contingency $5,000. "
                "Travel (5%, $13,750): OFC 2026 (conference presentation), "
                "CLEO 2026, NSF site visit. "
                "Indirect costs (10%, $27,500): UCLA negotiated indirect rate on "
                "modified total direct costs. "
                "Total: $275,000. Phase II ($1.75M) will scale to full prototype "
                "and independent testing at a DoD-designated test facility."
            ),
        ))

        return outline


# ── key grant writing rules ───────────────────────────────────────────

GRANT_WRITING_RULES = [
    ("Lead with the problem, not your solution",
     "Reviewers must CARE before they can FUND. Open every section with the "
     "unmet need, the cost of not solving it, or the market size. Do not open "
     "with 'We propose to...' -- open with 'X billion people lack Y because Z.'"),
    ("One sentence, one idea",
     "Federal reviewers read 50+ proposals. Dense paragraphs lose. "
     "Aim for Flesch reading ease > 50. Short sentences. Active voice. "
     "State results first, methods second."),
    ("Make every milestone falsifiable",
     "A milestone like 'investigate methods' cannot be evaluated. "
     "Write 'Achieve BER < 1e-3 at 10 Gbit/s by Month 4.' "
     "Reviewers reward concrete success criteria."),
    ("The budget tells a story",
     "Every dollar must trace to a technical task. "
     "Never write 'supplies $5,000' -- write 'CFBG module $800, "
     "fiber connectors $200, oscilloscope probes $300, contingency $3,700.' "
     "Reviewers kill vague budgets."),
    ("Prior work is not bragging, it is proof",
     "List prior publications, code repos, and demos. "
     "An SBIR Phase I reviewer's first question is: can this team actually do it? "
     "A GitHub repo with 35 tested modules is stronger evidence than a CV line."),
    ("Cite the program solicitation back at reviewers",
     "Use the exact words from the BAA or solicitation in your text. "
     "If the solicitation says 'next-generation photonic sensing', "
     "your Significance section says 'next-generation photonic sensing'. "
     "This is not plagiarism -- it is alignment."),
    ("Include a risk table",
     "A risk table (Technical Risk | Mitigation | Fallback) shows maturity. "
     "Example: 'GS does not converge for QAM-16 | reduce constellation size | "
     "fall back to BPSK and re-submit Phase II.' Reviewers reward honesty."),
]


def print_grant_rules():
    """Print the 7 grant writing rules in a readable format."""
    print("\n=== 7 RULES FOR WINNING A TIER-2 GRANT ===\n")
    for i, (rule, detail) in enumerate(GRANT_WRITING_RULES, 1):
        print(f"[{i}] {rule}")
        for line in textwrap.wrap(detail, width=70):
            print(f"    {line}")
        print()


def probability_of_funding(p_per_submission, n_submissions):
    """P(at least one award in n submissions) using geometric model.

    Federal grant success is a Bernoulli process: each submission is
    independent with probability p. The probability of winning at least
    once in n tries is 1 - (1-p)^n.
    """
    from dgs.probability_combinatorics import prob_success_within_n
    return {
        "p_per_submission": p_per_submission,
        "n_submissions": n_submissions,
        "p_at_least_one_award": prob_success_within_n(p_per_submission, n_submissions),
        "expected_submissions_to_win": 1.0 / p_per_submission,
    }


if __name__ == "__main__":
    print_grant_rules()

    print("\n=== SBIR Phase I acceptance rate (15%) -- how many submissions? ===")
    for n in [1, 3, 5, 10]:
        r = probability_of_funding(0.15, n)
        print(f"  n={n}: P(win) = {r['p_at_least_one_award']:.1%}")

    print("\n=== Generating SBIR Phase I proposal outline ===")
    outline = ProposalOutline.sbir_phase1_gs_receiver()
    outline.print_all()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    outline.save_json(path)
    print(f"JSON saved to {path}")
    outline2 = ProposalOutline.load_json(path)
    print(f"Reload check: {len(outline2.sections)} sections")
    os.unlink(path)
