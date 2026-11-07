# ABET write-up: dispersion-assisted phase recovery and the lab-desk simulator

Colin Casey. Continues the ECE 279AS project with Gabriel Morozowsky (Prof. Bahram Jalali, UCLA, 2024).

## What the project is

A photodiode only records intensity, so the phase of an optical pulse is lost. We send the pulse
down two fibers with different dispersion, record both intensity traces, and run time-domain
Gerchberg-Saxton (TDGSA) to get the phase back. In 2024 our MATLAB version didn't converge. This year I
found out why, fixed the model, and built a desk-style simulator of the bench that runs the algorithm
live, checks it two independent ways, and hands the data to MATLAB and Jupyter.

## How it meets the ABET student outcomes

**1. Solving a complex engineering problem.** The 2024 run stalled at roughly 99 % error and we didn't
know why. Treating it as a near-field versus far-field question fixed it: at 353 and 872 ps/nm both
photodiodes sit deep in the far field and record almost the same trace, so the algorithm has nothing
to work with. With 1 and 40 ps/nm the field comes back to about 7 % error. I also corrected the
ps/nm-to-β₂L unit conversion, which the 2024 slides flagged as a suspect.

**2. Design under constraints.** The constraint that matters is the detector. A real scope has tens of
GHz of bandwidth, and at 8 GHz the recovery fails (97 % error). The fix I modelled is a photonic
time-stretch front end: stretching the signal 30 times before the scope gives an 8 GHz scope 240 GHz
of effective bandwidth and brings the error down to 16 %; a 33 GHz scope stretched 30 times (about
1 THz) gets to 9 %. It is an idealised model (linear stretch, no dispersion penalty), and I say so in
the code.

**3. Communication.** The midterm deck, a README with numbered steps for opening everything, plain
docstrings at the top of each file, and the simulator itself, which shows the idea to someone who
won't read the equations: put the probes on the photodiodes, press Run, watch the red curve land on
the black one.

**4. Ethics and professional responsibility.** Two parts. First, honest reporting: the README and the
code state what is idealised (no detector noise by default, ideal stretch) and what hasn't been
verified on hardware; nothing here is a measurement. Second, several of the technology areas this work
touches (directed energy, integrated sensing) are dual-use. Characterising short pulses is mostly a
science and telecom problem, but I should be clear about who the work is for if it moves toward
high-power lasers.

**5. Teamwork.** The 2024 project was a two-person team; Gabriel led the algorithm slides and I worked on
the implementation. This year's work is solo, so I can't claim new evidence for this outcome beyond
building on and crediting the earlier joint work.

**6. Experimentation and judging data.** Every claim has a test (13 pass). Two of them are worth
pointing out. Causality: the gas-cell model the 2024 slides suspected puts 96 % of its response before
the light arrives; the corrected Kramers-Kronig model puts 0.09 % there. Topology: the recovered field
is checked against the true one by its phase vortices (the zeros of its time-frequency transform). A
good recovery puts 100 % of them in the right place; the 2024 settings put 0 %. That is a second,
independent way to judge a recovery, not just a single error number. MATLAB R2025b and Python agree
on the same capture (error 0.067 in both).

**7. Learning new material.** I had to pick up time-stretch and the dispersive Fourier transform, the
Kramers-Kronig relations, the topology of time-frequency zeros, and enough of pygame, MATLAB batch
mode and Jupyter tooling to tie it together.

## The DoD technology areas this fits

These are the areas arrowed on the OUSD(R&E) list. The fit is my reading, not an official one.

* **FutureG:** recovering phase from intensity alone is the idea behind receivers that don't need a
  local oscillator.
* **Integrated sensing:** single-shot gas spectroscopy through the gas cell.
* **Advanced computing and software:** the same algorithm in Python, MATLAB, C and CUDA (earlier work in
  this repo).
* **Trusted AI and autonomy:** the vortex check is the kind of physics-based test you would want on a
  neural-network reconstruction such as time-stretch spectral regression.
* **Directed energy:** characterising ultrafast pulses (see the dual-use note under outcome 4).
* **Human-machine interfaces:** the lab-desk simulator itself.

## Before submitting

* Check the outcome wording against your program's current ABET criteria and which outcomes the
  course actually assesses.
* The teamwork section is thin for new work; add a team component or say so plainly.
* References to cite: Gerchberg and Saxton, *Optik* 35 (1972); Solli, Gupta and Jalali, "Optical phase
  recovery in the dispersive Fourier transform," *Appl. Phys. Lett.* 95, 231108 (2009); and the
  "Neural network enabled time stretch spectral regression" paper. Check volumes, pages and authors
  against the PDFs.
