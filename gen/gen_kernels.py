#!/usr/bin/env python3
"""Generate AArch64 NEON kernels for ahnum (Apple Mach-O assembler syntax).

Data format: a complex "element" is one lane-group of 32 bytes,
    [re lane0, re lane1][im lane0, im lane1]
so each complex value occupies two vector registers (re, im), two lanes each.

FFT stage kernels (radix p in 2,3,4,5,7,8), all unnormalised:

  void ahn_dif_r{p}(double *base, size_t estride, size_t nlg, size_t nblocks,
                    size_t bstride, size_t s, const double *tw)
      for each block b, j < s, lane-group lg < nlg:
        elements t < p at base + b*bstride + j*estride + t*s*estride + lg*32
        y = DFT_p(x) (forward, e^{-2 pi i/p}); y_t *= tw[j][t-1]  (t >= 1)
      tw[j] holds p-1 packed complex [re, im] doubles.
  ahn_dif_r{p}_notw: same without twiddles (s == 1 or j == 0 only use).
  ahn_dif_r{p}_t4(base, estride, nlg, nblocks, bstride, delta):
      s == 1, no stage twiddles; y_t is multiplied element-wise by the complex
      vector stored at (address of y_t) + delta (four-step twiddle table).
  ahn_dit_r{p}, ahn_dit_r{p}_notw, ahn_dit_r{p}_t4: inverse: inputs are first
      multiplied by conj(twiddle), then inverse DFT_p.

Usage: gen_kernels.py OUTDIR
"""
import math
import os
import struct
import sys

PREFIX = "_ahn_"


def dbl(x):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


class Gen:
    def __init__(self):
        self.out = []
        self.consts = []

    def emit(self, s):
        self.out.append("\t" + s)

    def label(self, s):
        self.out.append(s + ":")


class Regs:
    """Very small vector register allocator."""

    def __init__(self, avail):
        self.free = list(avail)

    def get(self):
        if not self.free:
            raise RuntimeError("out of vector registers")
        return self.free.pop(0)

    def put(self, r):
        assert r not in self.free
        self.free.append(r)

    def cx(self):
        return [self.get(), self.get()]

    def putcx(self, c):
        self.put(c[0])
        self.put(c[1])


def v(r):
    return "v%d.2d" % r


def q(r):
    return "q%d" % r


class Fft:
    def __init__(self, g, regs, sign):
        self.g, self.R, self.sign = g, regs, sign  # sign -1 forward, +1 inverse

    def e(self, s):
        self.g.emit(s)

    def bfly(self, a, b):
        """(a, b) -> (a+b, a-b); sum stays in a's registers."""
        t = self.R.cx()
        for k in range(2):
            self.e("fsub %s, %s, %s" % (v(t[k]), v(a[k]), v(b[k])))
            self.e("fadd %s, %s, %s" % (v(a[k]), v(a[k]), v(b[k])))
        self.R.putcx(b)
        return a, t

    def bfly_j(self, a, b):
        """(a, b) -> (a + J b, a - J b) with J = -i (forward) or +i (inverse)."""
        t = self.R.cx()
        if self.sign < 0:
            # a - i b = (ar + bi, ai - br); a + i b = (ar - bi, ai + br)
            self.e("fsub %s, %s, %s" % (v(t[0]), v(a[0]), v(b[1])))
            self.e("fadd %s, %s, %s" % (v(t[1]), v(a[1]), v(b[0])))
            self.e("fadd %s, %s, %s" % (v(a[0]), v(a[0]), v(b[1])))
            self.e("fsub %s, %s, %s" % (v(a[1]), v(a[1]), v(b[0])))
        else:
            self.e("fadd %s, %s, %s" % (v(t[0]), v(a[0]), v(b[1])))
            self.e("fsub %s, %s, %s" % (v(t[1]), v(a[1]), v(b[0])))
            self.e("fsub %s, %s, %s" % (v(a[0]), v(a[0]), v(b[1])))
            self.e("fadd %s, %s, %s" % (v(a[1]), v(a[1]), v(b[0])))
        self.R.putcx(b)
        return a, t

    def w8(self, b, power, creg):
        """b * W8^power (power 1 or 3), creg holds [c, -c], c = sqrt(1/2)."""
        t = self.R.cx()
        br, bi = v(b[0]), v(b[1])
        fwd = self.sign < 0
        if power == 1:
            if fwd:  # (1-i)c: ((br+bi), (bi-br)) c
                self.e("fadd %s, %s, %s" % (v(t[0]), br, bi))
                self.e("fsub %s, %s, %s" % (v(t[1]), bi, br))
            else:  # (1+i)c: ((br-bi), (br+bi)) c
                self.e("fsub %s, %s, %s" % (v(t[0]), br, bi))
                self.e("fadd %s, %s, %s" % (v(t[1]), br, bi))
            self.e("fmul %s, %s, v%d.d[0]" % (v(t[0]), v(t[0]), creg))
            self.e("fmul %s, %s, v%d.d[0]" % (v(t[1]), v(t[1]), creg))
        else:
            if fwd:  # (-1-i)c: ((bi-br) c, (br+bi) (-c))
                self.e("fsub %s, %s, %s" % (v(t[0]), bi, br))
                self.e("fadd %s, %s, %s" % (v(t[1]), br, bi))
                self.e("fmul %s, %s, v%d.d[0]" % (v(t[0]), v(t[0]), creg))
                self.e("fmul %s, %s, v%d.d[1]" % (v(t[1]), v(t[1]), creg))
            else:  # (-1+i)c: ((br+bi)(-c), (br-bi) c)
                self.e("fadd %s, %s, %s" % (v(t[0]), br, bi))
                self.e("fsub %s, %s, %s" % (v(t[1]), br, bi))
                self.e("fmul %s, %s, v%d.d[1]" % (v(t[0]), v(t[0]), creg))
                self.e("fmul %s, %s, v%d.d[0]" % (v(t[1]), v(t[1]), creg))
        self.R.putcx(b)
        return t

    def dft2(self, x):
        a, b = self.bfly(x[0], x[1])
        return [a, b]

    def dft4(self, x):
        s0, d0 = self.bfly(x[0], x[2])
        s1, d1 = self.bfly(x[1], x[3])
        X0, X2 = self.bfly(s0, s1)
        X1, X3 = self.bfly_j(d0, d1)
        return [X0, X1, X2, X3]

    def dft8(self, x, creg):
        a0, b0 = self.bfly(x[0], x[4])
        a1, b1 = self.bfly(x[1], x[5])
        a2, b2 = self.bfly(x[2], x[6])
        a3, b3 = self.bfly(x[3], x[7])
        b1 = self.w8(b1, 1, creg)
        b3 = self.w8(b3, 3, creg)
        s0, d0 = self.bfly(a0, a2)
        s1, d1 = self.bfly(a1, a3)
        X0, X4 = self.bfly(s0, s1)
        X2, X6 = self.bfly_j(d0, d1)
        sb0, db0 = self.bfly_j(b0, b2)
        sb1, db1 = self.bfly(b1, b3)
        X1, X5 = self.bfly(sb0, sb1)
        X3, X7 = self.bfly_j(db0, db1)
        return [X0, X1, X2, X3, X4, X5, X6, X7]

    def dft_odd(self, x, p, cregs, store):
        """Odd-p DFT; calls store(t, value) as soon as each output is ready
        (the callee owns and frees the value's registers)."""
        m = (p - 1) // 2
        S, D = {}, {}
        for u in range(1, m + 1):
            S[u], D[u] = self.bfly(x[u], x[p - u])

        def cconst(vv):  # cos(2 pi vv / p), vv in 1..m
            idx = vv - 1
            return "v%d.d[%d]" % (cregs[idx // 2], idx % 2)

        def sconst(vv):
            idx = m + vv - 1
            return "v%d.d[%d]" % (cregs[idx // 2], idx % 2)

        x0 = x[0]
        for t in range(1, m + 1):
            A = self.R.cx()
            B = self.R.cx()
            for k in range(2):
                self.e("mov %s, %s" % (v(A[k]).replace(".2d", ".16b"), v(x0[k]).replace(".2d", ".16b")))
            first_b = True
            for u in range(1, m + 1):
                vv = (t * u) % p
                cv = vv if vv <= m else p - vv
                for k in range(2):
                    self.e("fmla %s, %s, %s" % (v(A[k]), v(S[u][k]), cconst(cv)))
                neg = vv > m
                for k in range(2):
                    if first_b:
                        self.e("fmul %s, %s, %s" % (v(B[k]), v(D[u][k]), sconst(cv)))
                        if neg:
                            self.e("fneg %s, %s" % (v(B[k]), v(B[k])))
                    else:
                        op = "fmls" if neg else "fmla"
                        self.e("%s %s, %s, %s" % (op, v(B[k]), v(D[u][k]), sconst(cv)))
                first_b = False
            # forward: y_t = A - iB, y_{p-t} = A + iB ; inverse swapped
            Y1 = self.R.cx()
            self.e("fadd %s, %s, %s" % (v(Y1[0]), v(A[0]), v(B[1])))
            self.e("fsub %s, %s, %s" % (v(Y1[1]), v(A[1]), v(B[0])))
            self.e("fsub %s, %s, %s" % (v(A[0]), v(A[0]), v(B[1])))
            self.e("fadd %s, %s, %s" % (v(A[1]), v(A[1]), v(B[0])))
            self.R.putcx(B)
            if self.sign < 0:
                store(t, Y1)
                store(p - t, A)
            else:
                store(t, A)
                store(p - t, Y1)
        for u in range(1, m + 1):
            for k in range(2):
                self.e("fadd %s, %s, %s" % (v(x0[k]), v(x0[k]), v(S[u][k])))
            self.R.putcx(S[u])
            self.R.putcx(D[u])
        store(0, x0)


def cmul_tw(f, y, treg, conj):
    """y *= (packed scalar twiddle in treg) or its conjugate; returns new value."""
    t = f.R.cx()
    yr, yi = v(y[0]), v(y[1])
    if not conj:
        f.e("fmul %s, %s, v%d.d[0]" % (v(t[0]), yr, treg))
        f.e("fmls %s, %s, v%d.d[1]" % (v(t[0]), yi, treg))
        f.e("fmul %s, %s, v%d.d[1]" % (v(t[1]), yr, treg))
        f.e("fmla %s, %s, v%d.d[0]" % (v(t[1]), yi, treg))
    else:
        f.e("fmul %s, %s, v%d.d[0]" % (v(t[0]), yr, treg))
        f.e("fmla %s, %s, v%d.d[1]" % (v(t[0]), yi, treg))
        f.e("fmul %s, %s, v%d.d[0]" % (v(t[1]), yi, treg))
        f.e("fmls %s, %s, v%d.d[1]" % (v(t[1]), yr, treg))
    f.R.putcx(y)
    return t


def cmul_vec(f, y, w, conj):
    """y *= w (complex vector pair) or conj(w); returns new value; frees w."""
    t = f.R.cx()
    yr, yi, wr, wi = v(y[0]), v(y[1]), v(w[0]), v(w[1])
    if not conj:
        f.e("fmul %s, %s, %s" % (v(t[0]), yr, wr))
        f.e("fmls %s, %s, %s" % (v(t[0]), yi, wi))
        f.e("fmul %s, %s, %s" % (v(t[1]), yr, wi))
        f.e("fmla %s, %s, %s" % (v(t[1]), yi, wr))
    else:
        f.e("fmul %s, %s, %s" % (v(t[0]), yr, wr))
        f.e("fmla %s, %s, %s" % (v(t[0]), yi, wi))
        f.e("fmul %s, %s, %s" % (v(t[1]), yi, wr))
        f.e("fmls %s, %s, %s" % (v(t[1]), yr, wi))
    f.R.putcx(y)
    f.R.putcx(w)
    return t


SAVE = """\tstp d8, d9, [sp, #-128]!
\tstp d10, d11, [sp, #16]
\tstp d12, d13, [sp, #32]
\tstp d14, d15, [sp, #48]
\tstp x19, x20, [sp, #64]
\tstp x21, x22, [sp, #80]
\tstp x23, x24, [sp, #96]
\tstp x25, x26, [sp, #112]"""

RESTORE = """\tldp d10, d11, [sp, #16]
\tldp d12, d13, [sp, #32]
\tldp d14, d15, [sp, #48]
\tldp x19, x20, [sp, #64]
\tldp x21, x22, [sp, #80]
\tldp x23, x24, [sp, #96]
\tldp x25, x26, [sp, #112]
\tldp d8, d9, [sp], #128"""


def const_table(g, name, vals):
    g.consts.append((name, vals))


# Complex element layouts. 'a': lane-group of 32 bytes [re l0 l1][im l0 l1],
# lane-groups consecutive. 'b': 8-lane blocks of 128 bytes [re l0..l7][im l0..l7],
# lane-group g = lanes 2g, 2g+1 with re at 16g and im at 64 + 16g.
LAYOUT = {"a": (16, 32), "b": (64, 16)}  # (im offset, lane-group step)


def ldc(g, c, addr, layout):
    imoff = LAYOUT[layout][0]
    if imoff == 16:
        g.emit("ldp %s, %s, [%s]" % (q(c[0]), q(c[1]), addr))
    else:
        g.emit("ldr %s, [%s]" % (q(c[0]), addr))
        g.emit("ldr %s, [%s, #%d]" % (q(c[1]), addr, imoff))


def stc(g, c, addr, layout):
    imoff = LAYOUT[layout][0]
    if imoff == 16:
        g.emit("stp %s, %s, [%s]" % (q(c[0]), q(c[1]), addr))
    else:
        g.emit("str %s, [%s]" % (q(c[0]), addr))
        g.emit("str %s, [%s, #%d]" % (q(c[1]), addr, imoff))


def gen_stage(g, p, direction, mode, layout="a"):
    """direction 'dif' or 'dit'; mode 'tw', 'notw' or 't4'; layout 'a' or 'b'
    (layout 'b' kernels carry the suffix _b)."""
    name = "%s%s_r%d%s%s" % (PREFIX, direction, p, "" if mode == "tw" else "_" + mode,
                             "" if layout == "a" else "_b")
    sign = -1 if direction == "dif" else 1
    g.out.append("")
    g.out.append(".globl %s" % name)
    g.out.append(".p2align 5")
    g.label(name)
    g.out.append(SAVE)
    # x0 base, x1 estride, x2 nlg, x3 nblocks, x4 bstride, x5 s, x6 tw / delta
    if mode == "t4":
        g.emit("mov x6, x5")  # delta
        g.emit("mov x5, #1")
    g.emit("mul x7, x5, x1")  # S = s * estride
    # element offsets t*S in x19..x25
    offregs = ["xzr", "x7", "x19", "x20", "x21", "x22", "x23", "x24"]
    for t in range(2, p):
        g.emit("add %s, %s, x7" % (offregs[t], offregs[t - 1]))

    nconst = 0
    cname = None
    if p == 8:
        nconst = 1
        cname = "Lc8_%s_%s_%s" % (direction, mode, layout)
        c = math.sqrt(0.5)
        const_table(g, cname, [c, -c])
    elif p in (3, 5, 7):
        m = (p - 1) // 2
        vals = [math.cos(2 * math.pi * k / p) for k in range(1, m + 1)]
        vals += [math.sin(2 * math.pi * k / p) for k in range(1, m + 1)]
        if len(vals) % 2:
            vals.append(0.0)
        nconst = len(vals) // 2
        cname = "Lcp%d_%s_%s_%s" % (p, direction, mode, layout)
        const_table(g, cname, vals)
    # register budget: twiddles at the top, constants below them
    ntw = p - 1 if mode == "tw" else 0
    allregs = list(range(32))
    twregs = allregs[32 - ntw:]
    cregs = allregs[32 - ntw - nconst:32 - ntw]
    avail = allregs[:32 - ntw - nconst]
    if cname:
        g.emit("adrp x9, %s@PAGE" % cname)
        g.emit("add x9, x9, %s@PAGEOFF" % cname)
        for i in range(0, nconst):
            g.emit("ldr q%d, [x9, #%d]" % (cregs[i], 16 * i))

    g.emit("mov x8, x0")  # block pointer
    g.label("1")  # block loop
    g.emit("mov x9, x8")  # j pointer
    g.emit("mov x10, x6")  # twiddle pointer
    g.emit("mov x11, x5")  # j counter
    g.label("2")  # j loop
    if mode == "tw":
        for t in range(ntw):
            g.emit("ldr q%d, [x10], #16" % twregs[t])
    g.emit("mov x12, x9")  # lane-group pointer
    g.emit("mov x13, x2")  # lg counter
    g.label("3")  # lg loop

    R = Regs(avail)
    f = Fft(g, R, sign)
    x = []
    for t in range(p):
        c = R.cx()
        g.emit("add x14, x12, %s" % offregs[t])
        ldc(g, c, "x14", layout)
        if direction == "dit" and t > 0 and mode == "tw":
            c = cmul_tw(f, c, twregs[t - 1], conj=True)
        if direction == "dit" and mode == "t4":
            w = R.cx()
            g.emit("add x15, x14, x6")
            ldc(g, w, "x15", layout)
            c = cmul_vec(f, c, w, conj=True)
        x.append(c)

    def store(t, y):
        if direction == "dif" and t > 0 and mode == "tw":
            y = cmul_tw(f, y, twregs[t - 1], conj=False)
        g.emit("add x14, x12, %s" % offregs[t])
        if direction == "dif" and mode == "t4":
            w = R.cx()
            g.emit("add x15, x14, x6")
            ldc(g, w, "x15", layout)
            y = cmul_vec(f, y, w, conj=False)
        stc(g, y, "x14", layout)
        R.putcx(y)

    if p == 2:
        ys = f.dft2(x)
    elif p == 4:
        ys = f.dft4(x)
    elif p == 8:
        ys = f.dft8(x, cregs[0])
    else:
        ys = None
        f.dft_odd(x, p, cregs, store)
    if ys is not None:
        for t in range(p):
            store(t, ys[t])

    g.emit("add x12, x12, #%d" % LAYOUT[layout][1])
    g.emit("subs x13, x13, #1")
    g.emit("b.ne 3b")
    g.emit("add x9, x9, x1")
    g.emit("subs x11, x11, #1")
    g.emit("b.ne 2b")
    g.emit("add x8, x8, x4")
    g.emit("subs x3, x3, #1")
    g.emit("b.ne 1b")
    g.out.append(RESTORE)
    g.emit("ret")


def gen_norm(g, mode, table, err):
    """Pass-1 normalisation for one group, one row pair per iteration.

    double ahn_norm_{a,b,c}[t][e](double *gbase, size_t rowstride, size_t npairs,
                                  double *carry, const double *A, const double *IA,
                                  const uint64_t *mask, const double *consts)
    gbase points at the first row of the group: one 8-column block (layout b,
    [re c0..c7][im c0..c7]) per row.
    Row pair (r, r+1) is processed with lanes = rows. Per digit i < 16:
      a: x = rint(d * osc) + carry; carry = rint(x / base); x -= carry*base;
         d = x * w
      b: same but stores x (unweighted digit)
      c: input d is a digit; x = d + carry; digits 0..14 carry as above, digit
         15 absorbs; d = x * w
    Compressed weights (table = False): osc = IA*IBs[i]*2^e, w = A*B[i]*2^-e,
      base = 2^(b+big); mask bit i: e flag, bit 16+i: big digit.
      consts: B[16], IBs[16], {2^b,2^b}, {2^-b,2^-b}, {1<<52,1<<52}.
    Table weights (table = True, suffix t): A points at 16 records per row pair
      of {osc[2], w[2], 1/base[2], base[2]} (64 bytes); IA, mask, consts unused.
    err (suffix e): track and return the max rounding error, else return 0.
    """
    name = "%snorm_%s%s%s" % (PREFIX, mode, "t" if table else "", "e" if err else "")
    g.out.append("")
    g.out.append(".globl %s" % name)
    g.out.append(".p2align 5")
    g.label(name)
    g.out.append(SAVE)
    E = g.emit
    if not table:
        E("ldr q30, [x7, #256]")  # base0
        E("ldr q29, [x7, #272]")  # ibase0
        E("ldr q31, [x7, #288]")  # K52
    E("movi v28.2d, #0")      # max error
    E("add x9, x0, x1")       # second row pointer
    E("lsl x1, x1, #1")       # advance two rows per iteration
    g.label("1")
    if not table:
        E("ldr q27, [x4], #16")   # A
        E("ldr q26, [x5], #16")   # IA
        E("ldr q25, [x6], #16")   # masks
    E("ldr q24, [x3]")        # carry in
    for lg in range(4):
        if not table:
            E("ldp q20, q21, [x7, #%d]" % (32 * lg))        # B[4lg .. 4lg+3]
            E("ldp q22, q23, [x7, #%d]" % (128 + 32 * lg))  # IBs
        E("ldr q0, [x0, #%d]" % (16 * lg))       # row r: re (layout b)
        E("ldr q1, [x0, #%d]" % (64 + 16 * lg))  #        im
        E("ldr q2, [x9, #%d]" % (16 * lg))       # row r+1
        E("ldr q3, [x9, #%d]" % (64 + 16 * lg))
        E("trn1 v4.2d, v0.2d, v2.2d")  # digit 4lg+0
        E("trn1 v5.2d, v1.2d, v3.2d")  # digit 4lg+1
        E("trn2 v6.2d, v0.2d, v2.2d")  # digit 4lg+2
        E("trn2 v7.2d, v1.2d, v3.2d")  # digit 4lg+3
        for k in range(4):
            i = 4 * lg + k
            D = "v%d.2d" % (4 + k)
            last = mode == "c" and i == 15
            if table:
                # v10 osc, v17 w, v14 1/base, v15 base
                E("ldp q10, q17, [x4], #32")
                E("ldp q14, q15, [x4], #32")
            else:
                breg = "v%d.d[%d]" % (20 + k // 2, k % 2)
                ibreg = "v%d.d[%d]" % (22 + k // 2, k % 2)
                E("shl v8.2d, v25.2d, #%d" % (52 - i))
                E("and v8.16b, v8.16b, v31.16b")      # te
                if not last:
                    E("shl v9.2d, v25.2d, #%d" % (36 - i))
                    E("and v9.16b, v9.16b, v31.16b")  # tb
                    E("sub v14.2d, v29.2d, v9.2d")    # 1/base
                    E("add v15.2d, v30.2d, v9.2d")    # base
                if mode != "c":
                    E("fmul v10.2d, v26.2d, %s" % ibreg)
                    E("add v10.2d, v10.2d, v8.2d")    # osc
                if mode != "b":
                    E("fmul v17.2d, v27.2d, %s" % breg)
                    E("sub v17.2d, v17.2d, v8.2d")    # w
            if mode == "c":
                E("fadd v12.2d, %s, v24.2d" % D)
            else:
                E("fmul v11.2d, %s, v10.2d" % D)
                E("frintn v12.2d, v11.2d")
                if err:
                    E("fabd v13.2d, v11.2d, v12.2d")
                    E("fmax v28.2d, v28.2d, v13.2d")
                E("fadd v12.2d, v12.2d, v24.2d")
            if not last:
                E("fmul v16.2d, v12.2d, v14.2d")
                E("frintn v24.2d, v16.2d")         # carry out
                E("fmls v12.2d, v24.2d, v15.2d")
            if mode == "b":
                E("mov %s, v12.16b" % D.replace(".2d", ".16b"))
            else:
                E("fmul %s, v12.2d, v17.2d" % D)
        E("trn1 v0.2d, v4.2d, v6.2d")
        E("trn1 v1.2d, v5.2d, v7.2d")
        E("trn2 v2.2d, v4.2d, v6.2d")
        E("trn2 v3.2d, v5.2d, v7.2d")
        E("str q0, [x0, #%d]" % (16 * lg))
        E("str q1, [x0, #%d]" % (64 + 16 * lg))
        E("str q2, [x9, #%d]" % (16 * lg))
        E("str q3, [x9, #%d]" % (64 + 16 * lg))
    if mode == "c":
        E("movi v24.2d, #0")
    E("str q24, [x3], #16")
    E("add x0, x0, x1")
    E("add x9, x9, x1")
    E("subs x2, x2, #1")
    E("b.ne 1b")
    E("fmaxp d0, v28.2d")
    g.out.append(RESTORE)
    E("ret")


def gen_norm_quad(g, mode, err):
    """Table-driven normalisation, two row pairs (4 rows) per iteration with
    interleaved carry chains. Same ABI as ahn_norm_*t*; npairs must be even.
    Name: ahn_norm_{a,b,c}q[e]. Carry chain per digit: c' = rint(x/base + c/base)
    computed as one FMA + round."""
    name = "%snorm_%sq%s" % (PREFIX, mode, "e" if err else "")
    g.out.append("")
    g.out.append(".globl %s" % name)
    g.out.append(".p2align 5")
    g.label(name)
    g.out.append(SAVE)
    E = g.emit
    E("movi v30.2d, #0")       # max error
    E("add x9, x0, x1")        # row r+1
    E("add x10, x9, x1")       # row r+2
    E("add x11, x10, x1")      # row r+3
    E("lsl x1, x1, #2")        # advance four rows
    E("lsr x2, x2, #1")        # quads
    E("add x12, x4, #1024")    # table for the second row pair
    g.label("1")
    E("ldp q24, q25, [x3]")    # carries P (rows r, r+1), Q (rows r+2, r+3)
    pairs = [
        dict(rows=("x0", "x9"), ld=(0, 1, 2, 3), dig=(4, 5, 6, 7), tab=(8, 9, 10, 11),
             tp="x4", cin=24, t1=26, t2=27),
        dict(rows=("x10", "x11"), ld=(16, 17, 18, 19), dig=(20, 21, 22, 23),
             tab=(12, 13, 14, 15), tp="x12", cin=25, t1=28, t2=29),
    ]
    for lg in range(4):
        for P in pairs:
            a, b, c, d = P["ld"]
            E("ldr q%d, [%s, #%d]" % (a, P["rows"][0], 16 * lg))
            E("ldr q%d, [%s, #%d]" % (b, P["rows"][0], 64 + 16 * lg))
            E("ldr q%d, [%s, #%d]" % (c, P["rows"][1], 16 * lg))
            E("ldr q%d, [%s, #%d]" % (d, P["rows"][1], 64 + 16 * lg))
            D = P["dig"]
            E("trn1 v%d.2d, v%d.2d, v%d.2d" % (D[0], a, c))
            E("trn1 v%d.2d, v%d.2d, v%d.2d" % (D[1], b, d))
            E("trn2 v%d.2d, v%d.2d, v%d.2d" % (D[2], a, c))
            E("trn2 v%d.2d, v%d.2d, v%d.2d" % (D[3], b, d))
        for k in range(4):
            i = 4 * lg + k
            last = mode == "c" and i == 15
            for P in pairs:
                osc, w, ib, base = P["tab"]
                Dk = P["dig"][k]
                cin, t1, t2 = P["cin"], P["t1"], P["t2"]
                E("ldp q%d, q%d, [%s], #32" % (osc, w, P["tp"]))
                E("ldp q%d, q%d, [%s], #32" % (ib, base, P["tp"]))
                if mode == "c":
                    E("mov v%d.16b, v%d.16b" % (t1, Dk))
                else:
                    E("fmul v%d.2d, v%d.2d, v%d.2d" % (Dk, Dk, osc))
                    E("frintn v%d.2d, v%d.2d" % (t1, Dk))
                    if err:
                        E("fabd v31.2d, v%d.2d, v%d.2d" % (Dk, t1))
                        E("fmax v30.2d, v30.2d, v31.2d")
                if last:
                    E("fadd v%d.2d, v%d.2d, v%d.2d" % (t1, t1, cin))
                    E("movi v%d.2d, #0" % cin)
                else:
                    E("fmul v%d.2d, v%d.2d, v%d.2d" % (t2, t1, ib))
                    E("fmla v%d.2d, v%d.2d, v%d.2d" % (t2, cin, ib))
                    E("fadd v%d.2d, v%d.2d, v%d.2d" % (t1, t1, cin))
                    E("frintn v%d.2d, v%d.2d" % (cin, t2))
                    E("fmls v%d.2d, v%d.2d, v%d.2d" % (t1, cin, base))
                if mode == "b":
                    E("mov v%d.16b, v%d.16b" % (Dk, t1))
                else:
                    E("fmul v%d.2d, v%d.2d, v%d.2d" % (Dk, t1, w))
        for P in pairs:
            a, b, c, d = P["ld"]
            D = P["dig"]
            E("trn1 v%d.2d, v%d.2d, v%d.2d" % (a, D[0], D[2]))
            E("trn1 v%d.2d, v%d.2d, v%d.2d" % (b, D[1], D[3]))
            E("trn2 v%d.2d, v%d.2d, v%d.2d" % (c, D[0], D[2]))
            E("trn2 v%d.2d, v%d.2d, v%d.2d" % (d, D[1], D[3]))
            E("str q%d, [%s, #%d]" % (a, P["rows"][0], 16 * lg))
            E("str q%d, [%s, #%d]" % (b, P["rows"][0], 64 + 16 * lg))
            E("str q%d, [%s, #%d]" % (c, P["rows"][1], 16 * lg))
            E("str q%d, [%s, #%d]" % (d, P["rows"][1], 64 + 16 * lg))
    E("stp q24, q25, [x3], #32")
    E("add x4, x4, #1024")
    E("add x12, x12, #1024")
    for r in ("x0", "x9", "x10", "x11"):
        E("add %s, %s, x1" % (r, r))
    E("subs x2, x2, #1")
    E("b.ne 1b")
    E("fmaxp d0, v30.2d")
    g.out.append(RESTORE)
    E("ret")


def gen_rows(g):
    """void ahn_rows_in(const double *rp, const double *rpp, double *scr, size_t C)
       void ahn_rows_out(double *rp, double *rpp, const double *scr, size_t C)
    Transpose between two rows in layout b (8-column blocks [re c0..c7][im c0..c7])
    and the pass-2 scratch in layout a (element q = [re_p, re_pp][im_p, im_pp]).
    C must be a multiple of 8."""
    E = g.emit
    g.out.append("")
    g.out.append(".globl %srows_in" % PREFIX)
    g.out.append(".p2align 5")
    g.label(PREFIX + "rows_in")
    E("lsr x3, x3, #3")
    g.label("1")
    for h in range(2):  # half-block: column pairs 2h, 2h+1
        E("ldp q0, q1, [x0, #%d]" % (32 * h))       # re p: cols 4h..4h+3
        E("ldp q2, q3, [x0, #%d]" % (64 + 32 * h))  # im p
        E("ldp q4, q5, [x1, #%d]" % (32 * h))       # re pp
        E("ldp q6, q7, [x1, #%d]" % (64 + 32 * h))  # im pp
        for m in range(2):
            re_p, im_p, re_q, im_q = m, 2 + m, 4 + m, 6 + m
            E("trn1 v16.2d, v%d.2d, v%d.2d" % (re_p, re_q))
            E("trn1 v17.2d, v%d.2d, v%d.2d" % (im_p, im_q))
            E("trn2 v18.2d, v%d.2d, v%d.2d" % (re_p, re_q))
            E("trn2 v19.2d, v%d.2d, v%d.2d" % (im_p, im_q))
            E("stp q16, q17, [x2], #32")
            E("stp q18, q19, [x2], #32")
    E("add x0, x0, #128")
    E("add x1, x1, #128")
    E("subs x3, x3, #1")
    E("b.ne 1b")
    E("ret")
    g.out.append("")
    g.out.append(".globl %srows_out" % PREFIX)
    g.out.append(".p2align 5")
    g.label(PREFIX + "rows_out")
    E("lsr x3, x3, #3")
    g.label("1")
    for h in range(2):
        for m in range(2):
            E("ldp q16, q17, [x2], #32")
            E("ldp q18, q19, [x2], #32")
            E("trn1 v%d.2d, v16.2d, v18.2d" % m)        # re p
            E("trn1 v%d.2d, v17.2d, v19.2d" % (2 + m))  # im p
            E("trn2 v%d.2d, v16.2d, v18.2d" % (4 + m))  # re pp
            E("trn2 v%d.2d, v17.2d, v19.2d" % (6 + m))  # im pp
        E("stp q0, q1, [x0, #%d]" % (32 * h))
        E("stp q2, q3, [x0, #%d]" % (64 + 32 * h))
        E("stp q4, q5, [x1, #%d]" % (32 * h))
        E("stp q6, q7, [x1, #%d]" % (64 + 32 * h))
    E("add x0, x0, #128")
    E("add x1, x1, #128")
    E("subs x3, x3, #1")
    E("b.ne 1b")
    E("ret")


def gen_split_sq(g):
    """void ahn_split_sq(double *scr, size_t C, const double *rs, const double *cs)
    General row pair (lanes p, p'): lane l at q pairs with lane 1-l at C-1-q.
    rs = {rsr_p, rsr_p', rsi_p, rsi_p'} (w_H^{k_r} per lane), cs[q] = packed
    w_C^{k_c(q)}. Squares in the frequency domain (see neon.c split1)."""
    E = g.emit
    name = PREFIX + "split_sq"
    g.out.append("")
    g.out.append(".globl %s" % name)
    g.out.append(".p2align 5")
    g.label(name)
    g.out.append(SAVE)
    E("ldp q30, q31, [x2]")         # RSr, RSi
    E("add x9, x0, x1, lsl #5")     # one past the end
    E("sub x9, x9, #32")            # element C-1
    E("lsr x1, x1, #1")             # C/2 iterations
    g.label("1")
    E("ldr q29, [x3], #16")         # cs[q]
    E("ldp q0, q1, [x0]")           # A
    E("ldp q2, q3, [x9]")           # B (unswapped)
    E("ext v2.16b, v2.16b, v2.16b, #8")
    E("ext v3.16b, v3.16b, v3.16b, #8")
    # w2 = RS * cs
    E("fmul v4.2d, v30.2d, v29.d[0]")
    E("fmls v4.2d, v31.2d, v29.d[1]")   # w2r
    E("fmul v5.2d, v30.2d, v29.d[1]")
    E("fmla v5.2d, v31.2d, v29.d[0]")   # w2i
    E("fadd v6.2d, v0.2d, v2.2d")       # er
    E("fsub v7.2d, v1.2d, v3.2d")       # ei
    E("fsub v16.2d, v0.2d, v2.2d")      # dr
    E("fadd v17.2d, v1.2d, v3.2d")      # di
    E("fmul v18.2d, v6.2d, v6.2d")
    E("fmls v18.2d, v7.2d, v7.2d")      # e2r
    E("fmul v19.2d, v6.2d, v7.2d")
    E("fadd v19.2d, v19.2d, v19.2d")    # e2i
    E("fmul v20.2d, v17.2d, v17.2d")
    E("fmls v20.2d, v16.2d, v16.2d")    # o2r
    E("fmul v21.2d, v17.2d, v16.2d")
    E("fadd v21.2d, v21.2d, v21.2d")    # t2 = -o2i
    E("fmla v18.2d, v4.2d, v20.2d")
    E("fmla v18.2d, v5.2d, v21.2d")     # Xr
    E("fmls v19.2d, v4.2d, v21.2d")
    E("fmla v19.2d, v5.2d, v20.2d")     # Xi
    E("fmul v22.2d, v6.2d, v17.2d")
    E("fmla v22.2d, v7.2d, v16.2d")
    E("fadd v22.2d, v22.2d, v22.2d")    # Yr
    E("fmul v23.2d, v7.2d, v17.2d")
    E("fmls v23.2d, v6.2d, v16.2d")
    E("fadd v23.2d, v23.2d, v23.2d")    # Yi
    E("fsub v0.2d, v18.2d, v23.2d")
    E("fadd v1.2d, v19.2d, v22.2d")
    E("fadd v2.2d, v18.2d, v23.2d")
    E("fsub v3.2d, v22.2d, v19.2d")
    E("ext v2.16b, v2.16b, v2.16b, #8")
    E("ext v3.16b, v3.16b, v3.16b, #8")
    E("stp q0, q1, [x0], #32")
    E("stp q2, q3, [x9]")
    E("sub x9, x9, #32")
    E("subs x1, x1, #1")
    E("b.ne 1b")
    g.out.append(RESTORE)
    E("ret")


def gen_cmul_b(g, conj):
    """void ahn_cmul{c}_b(double *base, size_t estride, size_t n, ptrdiff_t delta)
    Multiply n layout-b elements (8 lanes) at base + i*estride by the complex
    values at the same address + delta (conjugated for the 'c' variant)."""
    E = g.emit
    name = "%scmul%s_b" % (PREFIX, "c" if conj else "")
    g.out.append("")
    g.out.append(".globl %s" % name)
    g.out.append(".p2align 5")
    g.label(name)
    g.label("1")
    E("add x9, x0, x3")
    for lg in range(4):
        E("ldr q0, [x0, #%d]" % (16 * lg))
        E("ldr q1, [x0, #%d]" % (64 + 16 * lg))
        E("ldr q2, [x9, #%d]" % (16 * lg))
        E("ldr q3, [x9, #%d]" % (64 + 16 * lg))
        E("fmul v4.2d, v0.2d, v2.2d")
        E("fmul v5.2d, v1.2d, v2.2d")
        if not conj:
            E("fmls v4.2d, v1.2d, v3.2d")
            E("fmla v5.2d, v0.2d, v3.2d")
        else:
            E("fmla v4.2d, v1.2d, v3.2d")
            E("fmls v5.2d, v0.2d, v3.2d")
        E("str q4, [x0, #%d]" % (16 * lg))
        E("str q5, [x0, #%d]" % (64 + 16 * lg))
    E("add x0, x0, x1")
    E("subs x2, x2, #1")
    E("b.ne 1b")
    E("ret")


def gen_sme_r8(g):
    """Radix-8 FFT stage on the SME unit (FP64 outer products), layout b.

    void ahn_sme_r8(double *base, size_t estride, size_t nblocks, size_t bstride,
                    size_t s, const double *mats)
    Items are (block b, position j); item elements u < 8 are 8-lane blocks
    (128 bytes: re[8], im[8]) at base + b*bstride + j*estride + u*s*estride.
    Each item is replaced by Y = M_j X with M_j a complex 8x8 matrix stored as
    16 vectors of 8 doubles: for u = 0..7, column u real parts then imaginary
    parts (1 KB per j). Twiddles are folded into M_j, so the same kernel does
    DIF stages (M = diag(w_j) F8) and DIT stages (M = conj(F8) diag(conj w_j)).
    nblocks*s must be even. Two items are in flight (tiles za0-za3 or za4-za7,
    alternating) to cover the FMOPA latency.
    """
    E = g.emit
    name = PREFIX + "sme_r8"
    g.out.append("")
    g.out.append(".globl %s" % name)
    g.out.append(".p2align 5")
    g.label(name)
    g.out.append(SAVE)
    # args: x0 base, x1 estride, x2 nblocks, x3 bstride, x4 s, x5 mats;
    # shift to x3 nblocks, x4 bstride, x5 s, x6 mats
    E("mov x6, x5")
    E("mov x5, x4")
    E("mov x4, x3")
    E("mov x3, x2")
    E("smstart")
    E("ptrue p0.d")
    E("mul x7, x5, x1")          # S = s*estride
    offs = ["xzr", "x7", "x19", "x20", "x21", "x22", "x23", "x24"]
    for u in range(2, 8):
        E("add %s, %s, x7" % (offs[u], offs[u - 1]))
    E("mov x25, #8")             # +64 bytes in st1d [x, x25, lsl #3]
    E("mul x26, x3, x5")         # items = nblocks * s
    E("lsr x26, x26, #1")        # pairs
    # item iterator: x9 = block base, x10 = j, x11 = element ptr, x12 = matrix ptr
    E("mov x9, x0")
    E("mov x10, #0")
    E("mov x11, x0")
    E("mov x12, x6")

    def next_item():
        # advance (b, j): j++, ptr += estride, mat += 1024; wrap at s
        E("add x10, x10, #1")
        E("add x11, x11, x1")
        E("add x12, x12, #1024")
        E("cmp x10, x5")
        E("b.ne 7f")
        E("mov x10, #0")
        E("add x9, x9, x4")
        E("mov x11, x9")
        E("mov x12, x6")
        g.label("7")

    def pair(tset):
        ta = [4 * tset + i for i in range(4)]  # A re, A im, B re, B im
        # A = current item (x11, x12); B = next item (x13, x14)
        E("mov x13, x11")
        E("mov x14, x12")
        next_item()
        E("mov x16, x11")
        E("mov x15, x12")
        next_item()
        # item A: x13 elements, x14 matrix; item B: x16 elements, x15 matrix
        # (x14 and x15 become the tile-slice indices for the stores)
        E("zero {za%d.d, za%d.d, za%d.d, za%d.d}" % tuple(ta))
        for u in range(8):
            zb = 8 * (u % 4)
            xa, xi_a, xb, xi_b, fa, fia, fb, fib = [zb + i for i in range(8)]
            E("add x17, x13, %s" % offs[u])
            E("ld1d {z%d.d}, p0/z, [x17]" % xa)
            E("ld1d {z%d.d}, p0/z, [x17, #1, mul vl]" % xi_a)
            E("add x17, x16, %s" % offs[u])
            E("ld1d {z%d.d}, p0/z, [x17]" % xb)
            E("ld1d {z%d.d}, p0/z, [x17, #1, mul vl]" % xi_b)
            E("ld1d {z%d.d}, p0/z, [x14, #%d, mul vl]" % (fa, (2 * u) % 8) if 2 * u < 8
              else "ld1d {z%d.d}, p0/z, [x14, #%d, mul vl]" % (fa, 2 * u - 8))
            E("ld1d {z%d.d}, p0/z, [x14, #%d, mul vl]" % (fia, (2 * u + 1) % 8) if 2 * u + 1 < 8
              else "ld1d {z%d.d}, p0/z, [x14, #%d, mul vl]" % (fia, 2 * u + 1 - 8))
            E("ld1d {z%d.d}, p0/z, [x15, #%d, mul vl]" % (fb, 2 * u if 2 * u < 8 else 2 * u - 8))
            E("ld1d {z%d.d}, p0/z, [x15, #%d, mul vl]" % (fib, 2 * u + 1 if 2 * u + 1 < 8 else 2 * u + 1 - 8))
            if u == 3:
                # columns 4..7 of the matrix are the next 512 bytes
                E("add x14, x14, #512")
                E("add x15, x15, #512")
            # each tile is touched every fourth FMOPA (latency ~4.6 cycles)
            E("fmopa za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[0], fa, xa))
            E("fmopa za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[2], fb, xb))
            E("fmopa za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[1], fa, xi_a))
            E("fmopa za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[3], fb, xi_b))
            E("fmops za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[0], fia, xi_a))
            E("fmops za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[2], fib, xi_b))
            E("fmopa za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[1], fia, xa))
            E("fmopa za%d.d, p0/m, p0/m, z%d.d, z%d.d" % (ta[3], fib, xb))
        # Read each tile out with two 4-row MOVAs (tile-slice stores are slow,
        # ~7.7 cycles each) and store the rows as vectors. x14, A's matrix
        # pointer, is free by now and serves as the slice index (w12..w15 only).
        E("mov w14, #0")
        E("mov w15, #4")
        for k, (tile, item, imag) in enumerate([(ta[0], "x13", 0), (ta[1], "x13", 1),
                                                (ta[2], "x16", 0), (ta[3], "x16", 1)]):
            zb = 8 * k
            E("mova {z%d.d-z%d.d}, za%dh.d[w14, 0:3]" % (zb, zb + 3, tile))
            E("mova {z%d.d-z%d.d}, za%dh.d[w15, 0:3]" % (zb + 4, zb + 7, tile))
            for t in range(8):
                E("add x17, %s, %s" % (item, offs[t]))
                if imag:
                    E("st1d {z%d.d}, p0, [x17, #1, mul vl]" % (zb + t))
                else:
                    E("st1d {z%d.d}, p0, [x17]" % (zb + t))

    g.label("1")
    pair(0)
    E("subs x26, x26, #1")
    E("b.eq 2f")
    pair(1)
    E("subs x26, x26, #1")
    E("b.ne 1b")
    g.label("2")
    E("smstop")
    g.out.append(RESTORE)
    E("ret")


def write(outdir, fname, g, arch=None):
    lines = ([arch] if arch else []) + [".text"] + g.out
    if g.consts:
        lines.append("")
        lines.append(".section __TEXT,__const")
        for name, vals in g.consts:
            lines.append(".p2align 4")
            lines.append(name + ":")
            for x in vals:
                lines.append("\t.quad %s" % dbl(x))
    with open(os.path.join(outdir, fname), "w") as fh:
        fh.write("// Generated by gen/gen_kernels.py -- do not edit.\n")
        fh.write("\n".join(lines) + "\n")


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    g = Gen()
    for p in (2, 3, 4, 5, 7, 8):
        for d in ("dif", "dit"):
            for mode in ("tw", "notw", "t4"):
                for layout in ("a", "b"):
                    gen_stage(g, p, d, mode, layout)
    write(outdir, "fft_stages.S", g)
    g = Gen()
    for mode in ("a", "b", "c"):
        for table in (False, True):
            for err in ((False, True) if mode != "c" else (False,)):
                gen_norm(g, mode, table, err)
        for err in ((False, True) if mode != "c" else (False,)):
            gen_norm_quad(g, mode, err)
    write(outdir, "norm.S", g)
    g = Gen()
    gen_rows(g)
    gen_split_sq(g)
    gen_cmul_b(g, False)
    gen_cmul_b(g, True)
    write(outdir, "pass2.S", g)
    g = Gen()
    gen_sme_r8(g)
    write(outdir, "sme.S", g, arch=".arch armv9-a\n.arch_extension sme2\n.arch_extension sme-f64f64")


if __name__ == "__main__":
    main()
