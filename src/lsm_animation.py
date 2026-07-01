"""
lsm_animation.py  —  Animazione Manim per Liquid State Machine (LSM)

Uso rapido (bassa qualità, preview):
    manim -pql lsm_animation.py TitleScene
    manim -pql lsm_animation.py ConnectivityScene
    manim -pql lsm_animation.py ReservoirStructureScene
    manim -pql lsm_animation.py DynamicsScene
    manim -pql lsm_animation.py STDPScene
    manim -pql lsm_animation.py LSMFullAnimation   # tutto (no 3D)

Alta qualità:
    manim -pqh lsm_animation.py DynamicsScene
    manim -pqh lsm_animation.py LSMFullAnimation

Prerequisiti:  pip install manim numpy
"""

from manim import *
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Parametri simulazione  (ridotti per visualizzazione, fedeli al codice reale)
# ──────────────────────────────────────────────────────────────────────────────
N_LAYERS = 6
LAYER_H  = 3
LAYER_W  = 3
N        = N_LAYERS * LAYER_H * LAYER_W   # 54 neuroni
N_INPUTS = N_LAYERS                        # 6 canali input
T_STEPS  = 55                              # time steps

# ──────────────────────────────────────────────────────────────────────────────
#  Helper: build positions, weights, simulate (numpy, no torch/snntorch)
# ──────────────────────────────────────────────────────────────────────────────

def build_positions(n_layers=N_LAYERS, lh=LAYER_H, lw=LAYER_W):
    coords = []
    for z in range(n_layers):
        for y in range(lh):
            for x in range(lw):
                coords.append([float(z), float(y), float(x)])
    return np.array(coords)


def build_reservoir_weights(positions, C=0.9, sigma=2.0, inh_ratio=0.2, seed=42):
    rng  = np.random.default_rng(seed)
    Nn   = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    d2   = (diff ** 2).sum(axis=-1)
    prob = np.clip(C * np.exp(-d2 / sigma ** 2), 0, 1)
    np.fill_diagonal(prob, 0)
    mask = (rng.random((Nn, Nn)) < prob).astype(float)
    W    = rng.normal(0.08, 0.03, (Nn, Nn)) * mask
    n_inh    = int(Nn * inh_ratio)
    inh_idx  = rng.choice(Nn, n_inh, replace=False)
    W[inh_idx, :] *= -1.0
    is_inh = np.zeros(Nn, dtype=bool)
    is_inh[inh_idx] = True
    return W, is_inh, prob


def build_input_weights(n_layers=N_LAYERS, lh=LAYER_H, lw=LAYER_W,
                        strength=0.45, seed=42):
    rng  = np.random.default_rng(seed)
    Nn   = n_layers * lh * lw
    Win  = np.zeros((Nn, n_layers))
    for layer in range(n_layers):
        s, e = layer * lh * lw, (layer + 1) * lh * lw
        Win[s:e, layer] = strength * (0.7 + 0.3 * rng.random(lh * lw))
    return Win


def simulate_lsm(Win, Wlsm, input_spk, alpha=0.85, beta=0.80, th=0.75):
    """LIF semplificato (RSynaptic-like) senza snntorch."""
    Nn = Wlsm.shape[0]
    T  = input_spk.shape[0]
    mem = np.zeros(Nn); syn = np.zeros(Nn); spk = np.zeros(Nn)
    spk_rec = np.zeros((T, Nn)); mem_rec = np.zeros((T, Nn))
    for t in range(T):
        syn = alpha * syn + Win @ input_spk[t] + Wlsm @ spk
        mem = beta  * mem + syn
        spk = (mem >= th).astype(float)
        mem[spk > 0] = 0.0
        spk_rec[t] = spk; mem_rec[t] = mem
    return spk_rec, mem_rec


# ──────────────────────────────────────────────────────────────────────────────
#  Pre-simulazione (eseguita una volta all'import)
# ──────────────────────────────────────────────────────────────────────────────
positions          = build_positions()
Wlsm, is_inh, prob = build_reservoir_weights(positions)
Win                = build_input_weights()

rng_in = np.random.default_rng(7)
rates  = 0.07 + 0.15 * np.abs(np.sin(np.linspace(0, np.pi, N_INPUTS)))
input_spikes = np.array(
    [(rng_in.random(T_STEPS) < rates[ch]).astype(float) for ch in range(N_INPUTS)]
).T  # shape (T, N_INPUTS)

spk_rec, mem_rec = simulate_lsm(Win, Wlsm, input_spikes)


# ──────────────────────────────────────────────────────────────────────────────
#  Scene 1 — Titolo
# ──────────────────────────────────────────────────────────────────────────────
class TitleScene(Scene):
    def construct(self):
        title = Text("Liquid State Machine", font_size=64, weight=BOLD)
        title.set_color_by_gradient(BLUE, TEAL)

        sub = Text("Reservoir Computing con Neuroni Spiking",
                   font_size=30, color=GRAY)
        sub.next_to(title, DOWN, buff=0.45)

        bullet_items = [
            f"Reservoir 3D: {N_LAYERS} layer x {LAYER_H}x{LAYER_W} = {N} neuroni",
            "Connettivita' probabilistica:  P = C * exp(-D^2 / sigma^2)",
            "20% neuroni inibitori (pesi negativi)",
            "Neuroni RSynaptic  (snntorch)",
            "STDP asimmetrico online",
        ]
        bullets = VGroup(*[
            VGroup(
                Dot(radius=0.07, color=TEAL),
                Text(item, font_size=24, color=LIGHTER_GRAY),
            ).arrange(RIGHT, buff=0.2, aligned_edge=UP)
            for item in bullet_items
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        bullets.next_to(sub, DOWN, buff=0.6)

        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub))
        self.play(FadeIn(bullets, lag_ratio=0.3), run_time=1.8)
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, sub, bullets)))


# ──────────────────────────────────────────────────────────────────────────────
#  Scene 2 — Connettività probabilistica
# ──────────────────────────────────────────────────────────────────────────────
class ConnectivityScene(Scene):
    def construct(self):
        title = Text("Connettività Probabilistica", font_size=44, color=YELLOW)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP), run_time=0.5)

        # formula
        formula = MathTex(
            r"P(i \to j) = C \cdot e^{-D_{ij}^2 \,/\, \sigma^2}",
            font_size=44)
        pvals = MathTex(r"C = 1.0,\quad \sigma = 2.0", font_size=30, color=GRAY)
        formula.next_to(title, DOWN, buff=0.4)
        pvals.next_to(formula, DOWN, buff=0.25)
        self.play(Write(formula))
        self.play(FadeIn(pvals))
        self.wait(0.8)

        # ── grafico P(D) ──────────────────────────────────────────────────────
        ax = Axes(
            x_range=[0, 5.5, 1],
            y_range=[0, 1.1, 0.25],
            x_length=5.5, y_length=3.0,
            axis_config={"color": WHITE, "include_tip": True},
            x_axis_config={"numbers_to_include": range(6)},
            y_axis_config={"numbers_to_include": [0, 0.5, 1.0]},
        ).shift(DOWN * 1.3 + LEFT * 1.5)

        xl = ax.get_x_axis_label(r"D_{ij}", direction=RIGHT)
        yl = ax.get_y_axis_label(r"P(i\!\to\!j)", direction=UP)

        curve = ax.plot(lambda d: np.exp(-d ** 2 / 4.0),
                        x_range=[0, 5.5], color=BLUE, stroke_width=3)
        area  = ax.get_area(curve, x_range=[0, 2.0], color=BLUE, opacity=0.2)

        self.play(Create(ax), Write(xl), Write(yl))
        self.play(Create(curve), run_time=1.5)
        self.play(FadeIn(area))

        lbl = Text("Alta P\n(vicini)", font_size=17, color=BLUE)
        lbl.move_to(ax.c2p(0.7, 0.6))
        self.play(FadeIn(lbl))

        # ── heatmap matrice di prob (k×k) ────────────────────────────────────
        k       = 18
        ps      = prob[np.ix_(range(k), range(k))]
        cs      = 0.32
        grid    = VGroup(*[
            Square(side_length=cs,
                   fill_color=interpolate_color(BLACK, BLUE, float(ps[i, j])),
                   fill_opacity=1, stroke_width=0)
            .move_to(RIGHT * (j - k / 2) * cs + DOWN * (i - k / 2) * cs)
            for i in range(k) for j in range(k)
        ])
        grid.shift(RIGHT * 3.6 + DOWN * 1.3)

        grid_lbl = Text(f"Prob matrix ({k}×{k})", font_size=18, color=GRAY)
        grid_lbl.next_to(grid, UP, buff=0.15)

        self.play(FadeIn(grid, lag_ratio=0.005), run_time=1.5)
        self.play(Write(grid_lbl))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, formula, pvals, ax, xl, yl,
                                  curve, area, lbl, grid, grid_lbl)))


# ──────────────────────────────────────────────────────────────────────────────
#  Scene 3 — Struttura 3D del Reservoir  (ThreeDScene, standalone)
# ──────────────────────────────────────────────────────────────────────────────
class ReservoirStructureScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-55 * DEGREES)

        title_2d = Text("Struttura 3D del Reservoir", font_size=34)
        title_2d.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title_2d)
        self.play(Write(title_2d))

        scale  = np.array([1.1, 0.85, 0.85])
        offset = np.array([-(N_LAYERS - 1) / 2 * scale[0],
                           -(LAYER_H  - 1) / 2 * scale[1],
                           -(LAYER_W  - 1) / 2 * scale[2]])

        spheres = []
        for i, (z, y, x) in enumerate(positions):
            pos3 = np.array([z, y, x]) * scale + offset
            col  = RED if is_inh[i] else BLUE_D
            s    = Sphere(radius=0.18, color=col).move_to(pos3)
            s.set_opacity(0.85)
            spheres.append(s)

        # aggiungi layer per layer
        for layer in range(N_LAYERS):
            s_idx = layer * LAYER_H * LAYER_W
            e_idx = s_idx + LAYER_H * LAYER_W
            grp   = VGroup(*spheres[s_idx:e_idx])
            self.play(FadeIn(grp, shift=OUT * 0.3), run_time=0.35)

        # connessioni locali (dist < 2.0)
        conn_lines = VGroup()
        shown = 0
        for i in range(N):
            for j in range(N):
                if shown >= 80:
                    break
                if Wlsm[i, j] == 0:
                    continue
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist < 2.2:
                    p1  = (positions[i] * scale + offset).tolist()
                    p2  = (positions[j] * scale + offset).tolist()
                    col = RED_A if Wlsm[i, j] < 0 else BLUE_A
                    ln  = Line3D(start=p1, end=p2, color=col, stroke_width=1.2)
                    ln.set_opacity(0.45)
                    conn_lines.add(ln)
                    shown += 1

        self.play(Create(conn_lines), run_time=2)

        # legenda fixed
        legend = VGroup(
            VGroup(Dot(color=BLUE_D), Text(" Eccitatorio", font_size=18, color=BLUE_D))
            .arrange(RIGHT, buff=0.1),
            VGroup(Dot(color=RED),    Text(" Inibitorio",  font_size=18, color=RED))
            .arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        legend.to_corner(UR).shift(LEFT * 0.3)
        self.add_fixed_in_frame_mobjects(legend)
        self.play(FadeIn(legend))

        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(VGroup(*spheres, conn_lines, legend, title_2d)))


# ──────────────────────────────────────────────────────────────────────────────
#  Scene 4 — Dinamiche in tempo reale
# ──────────────────────────────────────────────────────────────────────────────
class DynamicsScene(Scene):
    def construct(self):
        title = Text("Dinamiche del Reservoir", font_size=40, color=GREEN)
        title.to_edge(UP)
        self.play(Write(title))

        # ── pannello input (sinistra) ─────────────────────────────────────────
        in_lbl = Text("Input\nSpike\nTrains", font_size=18, color=GREEN)
        in_lbl.move_to(LEFT * 6.0 + UP * 0.8)
        self.play(Write(in_lbl))

        in_rects = []
        for ch in range(N_INPUTS):
            y = 1.8 - ch * 0.65
            r = Rectangle(width=0.38, height=0.48,
                           fill_color=GREEN, fill_opacity=0.05,
                           stroke_color=GREEN, stroke_width=1.5)
            r.move_to(LEFT * 6.0 + UP * y)
            lbl = Text(f"ch{ch}", font_size=12, color=GREEN).next_to(r, LEFT, buff=0.1)
            self.add(r, lbl)
            in_rects.append(r)

        # frecce Win input→layer
        win_arrows = VGroup()
        for ch in range(N_INPUTS):
            in_y = 1.8 - ch * 0.65
            layer_x = -3.0 + ch * 1.1 + 0.35
            layer_y = 1.5
            arr = Arrow(
                start=LEFT * 5.6 + UP * in_y,
                end=np.array([layer_x, layer_y, 0]),
                stroke_width=1.2, max_tip_length_to_length_ratio=0.12,
                color=GREEN_A, buff=0.05
            )
            arr.set_opacity(0.3)
            win_arrows.add(arr)
        self.play(Create(win_arrows), run_time=1)

        # ── reservoir grid (centro) ───────────────────────────────────────────
        res_lbl = Text("Reservoir", font_size=20, color=BLUE)
        res_lbl.move_to(LEFT * 0.2 + UP * 2.7)
        self.play(Write(res_lbl))

        npc     = LAYER_H * LAYER_W   # neuroni per layer
        circles = []
        for n in range(N):
            layer = n // npc
            pin   = n  % npc
            row   = pin // LAYER_W
            col   = pin  % LAYER_W
            x = -3.0 + layer * 1.1 + col * 0.35
            y =  1.8 - row  * 0.35
            base = RED_D if is_inh[n] else BLUE_D
            c = Circle(radius=0.14, fill_color=base, fill_opacity=0.55,
                       stroke_color=base, stroke_width=1.5)
            c.move_to([x, y, 0])
            circles.append(c)

        self.play(FadeIn(VGroup(*circles)), run_time=0.8)

        # etichette layer
        for layer in range(N_LAYERS):
            lx = -3.0 + layer * 1.1 + 0.35
            lbl = Text(f"L{layer + 1}", font_size=12, color=GRAY)
            lbl.move_to([lx, 2.2, 0])
            self.add(lbl)

        # ── raster (destra) ───────────────────────────────────────────────────
        raster_lbl = Text("Raster", font_size=20, color=YELLOW)
        raster_lbl.move_to(RIGHT * 5.2 + UP * 2.7)
        self.play(Write(raster_lbl))

        raster_bg = Rectangle(width=2.2, height=3.8,
                               fill_color=BLACK, fill_opacity=0.5,
                               stroke_color=GRAY, stroke_width=1)
        raster_bg.move_to(RIGHT * 5.2 + DOWN * 0.3)
        self.add(raster_bg)

        # assi raster minimi
        ax_y_lbl = Text("neurone", font_size=11, color=GRAY).rotate(PI / 2)
        ax_y_lbl.next_to(raster_bg, LEFT, buff=0.05)
        ax_x_lbl = Text("tempo →", font_size=11, color=GRAY)
        ax_x_lbl.next_to(raster_bg, DOWN, buff=0.05)
        self.add(ax_y_lbl, ax_x_lbl)

        # ── membrane potential (un neurone campione) ──────────────────────────
        SAMPLE = 4
        mem_lbl = Text(f"Potenziale membrana  (neurone #{SAMPLE})",
                       font_size=17, color=WHITE)
        mem_lbl.move_to(LEFT * 1.0 + DOWN * 2.5)
        self.play(Write(mem_lbl))

        bar_bg = Rectangle(width=4.5, height=0.42,
                           fill_color=DARKER_GRAY, fill_opacity=0.8,
                           stroke_color=GRAY, stroke_width=1)
        bar_bg.move_to(LEFT * 1.0 + DOWN * 3.1)
        self.add(bar_bg)

        th_frac  = 0.75 / 1.5   # th / max_expected_mem
        th_line  = Line(
            bar_bg.get_left() + RIGHT * (4.5 * th_frac),
            bar_bg.get_right(),
            color=RED_B, stroke_width=2, stroke_opacity=0.8
        )
        th_lbl   = Text("th", font_size=13, color=RED_B)
        th_lbl.next_to(bar_bg.get_left() + RIGHT * (4.5 * th_frac), UP, buff=0.06)
        self.add(th_line, th_lbl)

        mem_fill = Rectangle(width=0.05, height=0.34,
                             fill_color=TEAL, fill_opacity=0.9, stroke_width=0)
        mem_fill.align_to(bar_bg, LEFT).align_to(bar_bg, DOWN).shift(UP * 0.04)
        self.add(mem_fill)

        # ── contatore tempo ───────────────────────────────────────────────────
        t_counter = always_redraw(lambda: Text("", font_size=1))  # placeholder
        step_lbl  = Text("t = 0", font_size=22, color=WHITE)
        step_lbl.move_to(LEFT * 5.5 + DOWN * 3.1)
        self.add(step_lbl)

        # ── loop animazione ───────────────────────────────────────────────────
        T_SHOW = min(T_STEPS, 45)

        for t in range(T_SHOW):
            anims = []

            # input
            for ch in range(N_INPUTS):
                alpha_ = 0.9 if input_spikes[t, ch] > 0 else 0.05
                anims.append(in_rects[ch].animate
                             .set_fill(GREEN, opacity=alpha_))
                if input_spikes[t, ch] > 0:
                    anims.append(win_arrows[ch].animate.set_opacity(0.9))
                else:
                    anims.append(win_arrows[ch].animate.set_opacity(0.15))

            # reservoir neurons
            for n in range(N):
                base  = RED_D if is_inh[n] else BLUE_D
                spiked = spk_rec[t, n] > 0
                if spiked:
                    anims.append(
                        circles[n].animate
                        .set_fill(YELLOW, opacity=1.0)
                        .set_stroke(YELLOW_A, width=2.5)
                    )
                else:
                    v   = float(np.clip(mem_rec[t, n] / 0.75, 0, 1))
                    col = interpolate_color(base, WHITE, v * 0.45)
                    anims.append(
                        circles[n].animate
                        .set_fill(col, opacity=0.5 + v * 0.4)
                        .set_stroke(base, width=1.5)
                    )

            # membrane bar
            mv    = float(np.clip(mem_rec[t, SAMPLE] / 1.5, 0, 1))
            bw    = max(0.06, mv * 4.5)
            bcol  = YELLOW if spk_rec[t, SAMPLE] > 0 else TEAL
            new_b = Rectangle(width=bw, height=0.34,
                               fill_color=bcol, fill_opacity=0.9, stroke_width=0)
            new_b.align_to(bar_bg, LEFT).align_to(bar_bg, DOWN).shift(UP * 0.04)
            anims.append(Transform(mem_fill, new_b))

            # raster dots (prime 28 righe visibili)
            t_frac = t / T_SHOW
            for n in range(min(N, 28)):
                if spk_rec[t, n] > 0:
                    rx = raster_bg.get_left()[0] + 0.12 + t_frac * 2.0
                    ry = raster_bg.get_top()[1]  - 0.12 - n * (3.56 / 28)
                    d  = Dot(radius=0.028, color=YELLOW, fill_opacity=0.85)
                    d.move_to([rx, ry, 0])
                    self.add(d)

            # step label
            new_lbl = Text(f"t = {t}", font_size=22, color=WHITE)
            new_lbl.move_to(step_lbl.get_center())
            anims.append(Transform(step_lbl, new_lbl))

            self.play(*anims, run_time=0.11)

        # ── statistiche finali ────────────────────────────────────────────────
        total   = int(spk_rec[:T_SHOW].sum())
        fr      = spk_rec[:T_SHOW].mean() * 100
        n_act   = int((spk_rec[:T_SHOW].sum(axis=0) > 0).sum())

        stats = VGroup(
            Text(f"Spike totali:        {total}", font_size=20, color=YELLOW),
            Text(f"Firing rate medio:   {fr:.1f} %", font_size=20, color=YELLOW),
            Text(f"Neuroni attivi:      {n_act} / {N}", font_size=20, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        stats.move_to(RIGHT * 5.2 + DOWN * 2.8)

        self.play(FadeIn(stats))
        self.wait(2.5)

        self.play(FadeOut(Group(*self.mobjects)))


# ──────────────────────────────────────────────────────────────────────────────
#  Scene 5 — STDP asimmetrico
# ──────────────────────────────────────────────────────────────────────────────
class STDPScene(Scene):
    def construct(self):
        title = Text("Asymmetric STDP  (Spike-Timing Dependent Plasticity)",
                     font_size=34, color=ORANGE)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP), run_time=0.5)

        fplus = MathTex(
            r"\Delta W^+ = A_+ \cdot e^{-\Delta t / \tau_+}",
            r"\quad \Delta t > 0\ \ (\text{pre prima di post})",
            font_size=32)
        fminus = MathTex(
            r"\Delta W^- = -A_- \cdot e^{\,\Delta t / \tau_-}",
            r"\quad \Delta t < 0\ \ (\text{post prima di pre})",
            font_size=32)
        params = MathTex(
            r"A_+ = 0.01,\;\; A_- = 0.0033,\;\;"
            r"\tau_+ = 20\,\text{ms},\;\; \tau_- = 60\,\text{ms}",
            font_size=26, color=GRAY)

        fplus.next_to(title, DOWN, buff=0.45)
        fminus.next_to(fplus,  DOWN, buff=0.35)
        params.next_to(fminus, DOWN, buff=0.3)

        self.play(Write(fplus),  run_time=1.1)
        self.play(Write(fminus), run_time=1.1)
        self.play(Write(params))
        self.wait(0.5)

        ax = Axes(
            x_range=[-85, 85, 20],
            y_range=[-0.005, 0.013, 0.005],
            x_length=8, y_length=3.6,
            axis_config={"color": WHITE, "include_tip": True},
            x_axis_config={"numbers_to_include": [-80, -40, 0, 40, 80]},
        ).shift(DOWN * 1.8)

        xl = ax.get_x_axis_label(r"\Delta t\ (\text{ms})", direction=RIGHT)
        yl = ax.get_y_axis_label(r"\Delta W", direction=UP)

        def stdp(dt):
            # Δt = t_post - t_pre
            if dt > 0:   # pre prima di post → potenziamento LTP
                return 0.01  * np.exp(-dt / 20.0)
            else:         # post prima di pre → depressione LTD
                return -0.0033 * np.exp( dt / 60.0)

        curve    = ax.plot(stdp, x_range=[-85, 85, 0.5],
                           use_smoothing=False, color=ORANGE, stroke_width=3)
        h_line   = ax.plot(lambda _: 0, x_range=[-85, 85],
                           color=GRAY, stroke_width=1.0)
        pot_area = ax.get_area(curve, x_range=[0,   85], color=GREEN, opacity=0.18)
        dep_area = ax.get_area(curve, x_range=[-85, 0],  color=RED,   opacity=0.18)

        lbl_ltp = Text("LTP\n(potenziamento)", font_size=18, color=GREEN)
        lbl_ltd = Text("LTD\n(depressione)",   font_size=18, color=RED)
        lbl_ltp.move_to(ax.c2p( 55, 0.007))
        lbl_ltd.move_to(ax.c2p(-55, -0.003))

        self.play(Create(ax), Write(xl), Write(yl))
        self.play(Create(h_line), FadeIn(pot_area), FadeIn(dep_area))
        self.play(Create(curve), run_time=2)
        self.play(Write(lbl_ltp), Write(lbl_ltd))
        self.wait(3)

        self.play(FadeOut(Group(*self.mobjects)))


# ──────────────────────────────────────────────────────────────────────────────
#  Scene 6 — Sommario finale
# ──────────────────────────────────────────────────────────────────────────────
class SummaryScene(Scene):
    def construct(self):
        title = Text("Liquid State Machine — Sommario", font_size=40)
        title.set_color_by_gradient(BLUE, GREEN)
        title.to_edge(UP)

        items = [
            ("1. Reservoir 3D",       f"{N_LAYERS} layer × {LAYER_H}×{LAYER_W} = {N} neuroni",       BLUE),
            ("2. Connettività",        "P = C · exp(−D² / σ²)  →  sparse locale",                     GREEN),
            ("3. Neuroni inibitori",   "20% dei neuroni hanno W < 0  (controllo eccitabilità)",        RED),
            ("4. Dinamica RSynaptic",  "Sinapsi + membrana → spike + reset  (snntorch)",               TEAL),
            ("5. STDP asimmetrico",    "LTP: τ₊=20ms  |  LTD: τ₋=60ms  (A₊ > A₋)",                   ORANGE),
            ("6. Lettura del liquid",  "Spike state → readout lineare → classificazione",              YELLOW),
        ]

        rows = VGroup()
        for bold, rest, col in items:
            row = VGroup(
                Text(bold, font_size=24, color=col, weight=BOLD),
                Text("  " + rest, font_size=22, color=LIGHTER_GRAY),
            ).arrange(RIGHT, buff=0.1, aligned_edge=UP)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        rows.next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.4)
        self.wait(3)

        end = Text("Fine", font_size=72, color=WHITE, weight=BOLD)
        self.play(FadeOut(VGroup(title, rows)))
        self.play(Write(end))
        self.wait(1.5)
        self.play(FadeOut(end))


# ──────────────────────────────────────────────────────────────────────────────
#  Scene 7 — Dati reali: audio → Gammatone → LSM addestrato → spike raster
#
#  Uso:   manim -pql lsm_animation.py RealDataScene
#         manim -pqh lsm_animation.py RealDataScene
# ──────────────────────────────────────────────────────────────────────────────
import os as _os, sys as _sys

_HERE  = _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd()
_ROOT  = _os.path.dirname(_HERE)

_AUDIO_PATH = _os.path.join(_ROOT, "data", "01.wav")
_LSM_FILTER = _os.path.join(_ROOT, "results",
               "stella_maris_pretrain_56channel_all_windows_final", "lsm_filter.pth")
_NPZ_PATH   = _os.path.join(_ROOT, "results",
               "stella_maris_pretrain_56channel_all_windows_final", "spikes", "SC_001_spikes.npz")

_N_CH    = 56   # canali gammatone (come da training)
_N_NEUR  = _N_CH * 3 * 3   # 504 neuroni nel reservoir
_N_RAST  = 48   # neuroni mostrati nel raster
_T_ANIM  = 200  # frame massimi da animare (subsampled)


class RealDataScene(Scene):
    """
    Pipeline completa su dati reali:
      01.wav  →  LPC  →  Gammatone 56ch  →  lsm_filter  →  spike raster

    Se il modello o l'audio non sono accessibili, cade in fallback NPZ.
    """

    # ── helpers statici ───────────────────────────────────────────────────────
    @staticmethod
    def _feat_to_rgba(feat, cmap="hot"):
        """feat (C, T) float → (C, T, 4) uint8 RGBA  (basse freq in basso)."""
        import matplotlib.cm as cm
        fn = (feat - feat.min()) / (feat.max() - feat.min() + 1e-12)
        fn = np.flipud(fn)
        return (cm.get_cmap(cmap)(fn) * 255).astype(np.uint8)

    @staticmethod
    def _spk_to_rgba(spk, n_show=48):
        """spk (T, N) float → (n_show, T, 4) uint8 RGBA."""
        T, N = spk.shape
        n = min(n_show, N)
        img = np.zeros((n, T, 4), dtype=np.uint8)
        img[:, :, 3] = 220  # sfondo quasi-nero opaco
        rows, cols = np.where(spk[:, :n].T > 0)   # rows=neurone, cols=tempo
        img[rows, cols] = [255, 210, 0, 255]
        return img

    @staticmethod
    def _load_pipeline():
        """Carica audio → LPC → Gammatone → LSM. Ritorna (wave, feat, spk)."""
        import torch, torchaudio
        if _HERE not in _sys.path:
            _sys.path.insert(0, _HERE)
        from preprocessing import lowpass_filter_torch, linear_predictive_analysis
        from features     import residuals_to_gammatone_energy
        from lsm          import LSM

        # 1. Audio (2 secondi, ricampionato a 16 kHz)
        wav, sr_orig = torchaudio.load(_AUDIO_PATH)
        if wav.ndim > 1:
            wav = wav.mean(0)
        sr_orig = int(sr_orig)
        target_sr = 16000
        if sr_orig != target_sr:
            wav = torchaudio.functional.resample(wav, sr_orig, target_sr)
        sr = target_sr
        wav = wav[:sr * 2].float()
        wav = lowpass_filter_torch(wav, sr=sr)

        # 2. LPC  (ritorna 3 valori: coeffs, residuals, preds)
        lpc_coeffs, residuals, _preds = linear_predictive_analysis(wav, sr=sr)
        # residuals: (n_frames, win_len)

        # 3. Gammatone 56 canali (f_max=4000 Hz come nel training)
        feat = residuals_to_gammatone_energy(
            residuals, sr=sr, n_filters=_N_CH, f_max=4000.0)
        feat_np = feat.numpy()                                         # (56, T)
        feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-12)

        # 4. Carica LSM addestrato
        ckpt = torch.load(_LSM_FILTER, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            lsm_obj = LSM(n_layers=_N_CH, layer_h=3, layer_w=3, use_stdp=False)
            lsm_obj.load_state_dict(ckpt)
        else:
            lsm_obj = ckpt
        lsm_obj.eval()

        inp = torch.tensor(feat_np, dtype=torch.float32).T.unsqueeze(1)  # (T,1,56)
        with torch.no_grad():
            spk_out = lsm_obj(inp, apply_stdp=False)                     # (T,1,504)
        spk_np = spk_out[:, 0, :].numpy()                                # (T, 504)

        return wav.numpy(), feat_np, spk_np, "LPC + Gammatone + lsm_filter.pth"

    @staticmethod
    def _load_fallback():
        """Fallback: spike da NPZ (SC_001), waveform da file audio se disponibile."""
        import torch, torchaudio
        # Spikes pre-calcolati
        npz  = np.load(_NPZ_PATH)
        wl   = int(npz["window_lengths"][0])
        spk_np   = npz["filter_raw"][:wl, :].astype(np.float32)       # (T, 504)
        feat_fake = np.random.default_rng(0).uniform(0, 1, (_N_CH, wl)).astype(np.float32)

        try:
            wav, sr = torchaudio.load(_AUDIO_PATH)
            wav_np  = wav.mean(0)[:int(sr)*2].numpy()
        except Exception:
            wav_np = np.zeros(32000, dtype=np.float32)

        return wav_np, feat_fake, spk_np, "spike pre-calcolati (SC_001, lsm_filter)"

    # ── costruzione scena ─────────────────────────────────────────────────────
    def construct(self):
        # Caricamento dati
        try:
            wave_np, feat_np, spk_np, src_str = self._load_pipeline()
        except Exception as e_pipe:
            print(f"[RealDataScene] Pipeline fallita ({e_pipe}), uso NPZ fallback")
            try:
                wave_np, feat_np, spk_np, src_str = self._load_fallback()
            except Exception as e_npz:
                print(f"[RealDataScene] NPZ fallito ({e_npz}), dati sintetici")
                wave_np  = np.sin(np.linspace(0, 40 * np.pi, 32000)) * 0.5
                feat_np  = spk_rec.mean(axis=0)[None, :].repeat(_N_CH, 0).astype(np.float32)
                spk_np   = spk_rec.astype(np.float32)
                src_str  = "dati sintetici"

        # Subsample temporale
        T_full = min(feat_np.shape[1], spk_np.shape[0])
        step   = max(1, T_full // _T_ANIM)
        feat   = feat_np[:, ::step]                 # (56, T_a)
        spk    = spk_np[::step, :]                  # (T_a, 504)
        T_a    = feat.shape[1]

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("LSM su dati reali  —  Stella Maris",
                     font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.15)
        src_lbl = Text(f"Fonte: {src_str}", font_size=14, color=GRAY)
        src_lbl.next_to(title, DOWN, buff=0.08)
        self.play(Write(title), FadeIn(src_lbl), run_time=1.0)

        # ── Geometrie pannelli ─────────────────────────────────────────────────
        PW, PH = 3.5, 2.6
        PY     = -0.6
        CX_WAV, CX_FEAT, CX_RAST = -5.0, -0.7, 4.0

        # ── Waveform ──────────────────────────────────────────────────────────
        wav_ax = Axes(
            x_range=[0, 1, 0.5],
            y_range=[-1.1, 1.1, 0.5],
            x_length=PW, y_length=PH,
            axis_config={"color": GRAY, "stroke_width": 1},
            tips=False,
        ).move_to([CX_WAV, PY, 0])

        ds      = max(1, len(wave_np) // 2000)
        wn      = wave_np[::ds]
        wn      = wn / (np.abs(wn).max() + 1e-9)
        xs      = np.linspace(0, 1, len(wn))
        wav_curve = wav_ax.plot_line_graph(
            x_values=xs, y_values=wn,
            line_color=TEAL, add_vertex_dots=False, stroke_width=1.2
        )
        wav_lbl = Text("01.wav  (2 s, 16 kHz)", font_size=15, color=TEAL)
        wav_lbl.next_to(wav_ax, DOWN, buff=0.12)

        self.play(Create(wav_ax), Write(wav_lbl), run_time=0.8)
        self.play(Create(wav_curve), run_time=1.2)

        # ── Freccia 1 ─────────────────────────────────────────────────────────
        arr1 = Arrow(
            wav_ax.get_right() + RIGHT * 0.08,
            wav_ax.get_right() + RIGHT * 1.0,
            color=GREEN, stroke_width=2.5, max_tip_length_to_length_ratio=0.15
        )
        lbl1 = Text("LPC\n+\nGamma-\ntone\n56ch", font_size=13, color=GREEN)
        lbl1.next_to(arr1, UP, buff=0.08)
        self.play(Create(arr1), Write(lbl1))

        # ── Feature heatmap ───────────────────────────────────────────────────
        feat_rgba = self._feat_to_rgba(feat, cmap="hot")       # (56, T_a, 4)
        feat_img  = ImageMobject(feat_rgba)
        feat_img.stretch_to_fit_width(PW).stretch_to_fit_height(PH)
        feat_img.move_to([CX_FEAT, PY, 0])

        feat_border = SurroundingRectangle(feat_img,
                       color=ORANGE, stroke_width=1.5, buff=0.01)
        feat_lbl = Text(f"Gammatone  56 ch × {T_a} frame",
                        font_size=15, color=ORANGE)
        feat_lbl.next_to(feat_img, DOWN, buff=0.12)
        freq_lbl = Text("freq ↑", font_size=13, color=GRAY).rotate(PI / 2)
        freq_lbl.next_to(feat_img, LEFT, buff=0.1)
        time_lbl = Text("tempo →", font_size=13, color=GRAY)
        time_lbl.next_to(feat_img, DOWN, buff=0.35)

        self.play(FadeIn(feat_img), Create(feat_border),
                  Write(feat_lbl), FadeIn(freq_lbl), FadeIn(time_lbl))

        # ── Freccia 2 ─────────────────────────────────────────────────────────
        arr2 = Arrow(
            feat_img.get_right() + RIGHT * 0.08,
            feat_img.get_right() + RIGHT * 1.0,
            color=BLUE, stroke_width=2.5, max_tip_length_to_length_ratio=0.15
        )
        lbl2 = Text("LSM\nfilter\n(504 n.)", font_size=13, color=BLUE)
        lbl2.next_to(arr2, UP, buff=0.08)
        self.play(Create(arr2), Write(lbl2))

        # ── Spike raster (ImageMobject) ───────────────────────────────────────
        spk_rgba = self._spk_to_rgba(spk[:T_a, :], n_show=_N_RAST)  # (48, T_a, 4)
        spk_img  = ImageMobject(spk_rgba)
        spk_img.stretch_to_fit_width(PW).stretch_to_fit_height(PH)
        spk_img.move_to([CX_RAST, PY, 0])

        spk_border = SurroundingRectangle(spk_img,
                      color=YELLOW_A, stroke_width=1.5, buff=0.01)
        spk_lbl = Text(f"Spike raster  {_N_RAST} neuroni × {T_a} frame",
                       font_size=15, color=YELLOW)
        spk_lbl.next_to(spk_img, DOWN, buff=0.12)
        neur_lbl = Text("neurone ↓", font_size=13, color=GRAY).rotate(PI / 2)
        neur_lbl.next_to(spk_img, LEFT, buff=0.1)

        self.play(FadeIn(spk_img), Create(spk_border),
                  Write(spk_lbl), FadeIn(neur_lbl))

        # ── Reveal animato (maschera scorrevole) + cursore ────────────────────
        progress = ValueTracker(0.0)

        # Maschera feature (copre la parte destra non ancora rivelata)
        def make_feat_mask():
            p  = progress.get_value()
            mw = max(0.002, (1.0 - p) * PW)
            mx = feat_img.get_left()[0] + p * PW + mw / 2
            return Rectangle(width=mw, height=PH + 0.06,
                             fill_color=BLACK, fill_opacity=1.0,
                             stroke_width=0).move_to([mx, PY, 0])

        # Maschera spike raster
        def make_spk_mask():
            p  = progress.get_value()
            mw = max(0.002, (1.0 - p) * PW)
            mx = spk_img.get_left()[0] + p * PW + mw / 2
            return Rectangle(width=mw, height=PH + 0.06,
                             fill_color=BLACK, fill_opacity=1.0,
                             stroke_width=0).move_to([mx, PY, 0])

        # Linea cursore verticale (su entrambi i pannelli)
        def make_cursor():
            p  = progress.get_value()
            # due linee separate unite come VGroup
            x1 = feat_img.get_left()[0] + p * PW
            x2 = spk_img.get_left()[0]  + p * PW
            l1 = Line([x1, feat_img.get_top()[1],  0],
                      [x1, feat_img.get_bottom()[1], 0],
                      color=WHITE, stroke_width=1.8, stroke_opacity=0.85)
            l2 = Line([x2, spk_img.get_top()[1],   0],
                      [x2, spk_img.get_bottom()[1],  0],
                      color=WHITE, stroke_width=1.8, stroke_opacity=0.85)
            return VGroup(l1, l2)

        # Label tempo corrente
        def make_t_lbl():
            t_now = int(progress.get_value() * T_a)
            return Text(f"t = {t_now:3d} / {T_a}",
                        font_size=17, color=WHITE
                        ).move_to([CX_RAST, spk_img.get_bottom()[1] - 0.28, 0])

        feat_mask = always_redraw(make_feat_mask)
        spk_mask  = always_redraw(make_spk_mask)
        cursor    = always_redraw(make_cursor)
        t_counter = always_redraw(make_t_lbl)

        self.add(feat_mask, spk_mask, cursor, t_counter)

        # Colorbar mini (legenda heatmap)
        cbar_data = np.linspace(0, 1, 32).reshape(32, 1)
        import matplotlib.cm as _cm
        cbar_rgba = (_cm.get_cmap("hot")(cbar_data) * 255).astype(np.uint8)  # (32,1,4)
        cbar_img  = ImageMobject(cbar_rgba)
        cbar_img.stretch_to_fit_height(PH * 0.8).stretch_to_fit_width(0.18)
        cbar_img.next_to(feat_img, RIGHT, buff=0.08)
        cbar_hi = Text("alto", font_size=10, color=GRAY).next_to(cbar_img, UP, buff=0.04)
        cbar_lo = Text("basso", font_size=10, color=GRAY).next_to(cbar_img, DOWN, buff=0.04)
        self.add(cbar_img, cbar_hi, cbar_lo)

        # Animazione reveal (8 secondi, rate lineare)
        self.play(progress.animate.set_value(1.0),
                  run_time=9.0, rate_func=linear)
        self.wait(0.5)

        # ── Statistiche finali ─────────────────────────────────────────────────
        spk_sub  = spk[:T_a, :_N_RAST]
        n_spikes = int((spk_sub > 0).sum())
        fr_pct   = float((spk_sub > 0).mean()) * 100
        n_silent = int(((spk_sub > 0).sum(axis=0) == 0).sum())

        stats = VGroup(
            Text(f"Spike totali ({_N_RAST} neuroni): {n_spikes}",
                 font_size=18, color=YELLOW),
            Text(f"Firing rate medio:  {fr_pct:.1f}%",
                 font_size=18, color=YELLOW),
            Text(f"Neuroni silenti:    {n_silent} / {_N_RAST}",
                 font_size=18, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        stats.to_edge(DOWN, buff=0.12)

        self.play(FadeIn(stats))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ──────────────────────────────────────────────────────────────────────────────
#  LSMFullAnimation — tutte le scene 2D in sequenza
#  (ReservoirStructureScene rimane separata perché è ThreeDScene)
# ──────────────────────────────────────────────────────────────────────────────
class LSMFullAnimation(Scene):
    """
    manim -pql lsm_animation.py LSMFullAnimation
    manim -pqh lsm_animation.py LSMFullAnimation
    """
    def construct(self):
        TitleScene.construct(self)
        ConnectivityScene.construct(self)
        DynamicsScene.construct(self)
        STDPScene.construct(self)
        SummaryScene.construct(self)
