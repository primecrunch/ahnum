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


def gen_stage(g, p, direction, mode):
    """direction 'dif' or 'dit'; mode 'tw', 'notw' or 't4'."""
    name = "%s%s_r%d%s" % (PREFIX, direction, p, "" if mode == "tw" else "_" + mode)
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
        cname = "Lc8_%s_%s" % (direction, mode)
        c = math.sqrt(0.5)
        const_table(g, cname, [c, -c])
    elif p in (3, 5, 7):
        m = (p - 1) // 2
        vals = [math.cos(2 * math.pi * k / p) for k in range(1, m + 1)]
        vals += [math.sin(2 * math.pi * k / p) for k in range(1, m + 1)]
        if len(vals) % 2:
            vals.append(0.0)
        nconst = len(vals) // 2
        cname = "Lcp%d_%s_%s" % (p, direction, mode)
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
        g.emit("ldp %s, %s, [x14]" % (q(c[0]), q(c[1])))
        if direction == "dit" and t > 0 and mode == "tw":
            c = cmul_tw(f, c, twregs[t - 1], conj=True)
        if direction == "dit" and mode == "t4":
            w = R.cx()
            g.emit("add x15, x14, x6")
            g.emit("ldp %s, %s, [x15]" % (q(w[0]), q(w[1])))
            c = cmul_vec(f, c, w, conj=True)
        x.append(c)

    def store(t, y):
        if direction == "dif" and t > 0 and mode == "tw":
            y = cmul_tw(f, y, twregs[t - 1], conj=False)
        g.emit("add x14, x12, %s" % offregs[t])
        if direction == "dif" and mode == "t4":
            w = R.cx()
            g.emit("add x15, x14, x6")
            g.emit("ldp %s, %s, [x15]" % (q(w[0]), q(w[1])))
            y = cmul_vec(f, y, w, conj=False)
        g.emit("stp %s, %s, [x14]" % (q(y[0]), q(y[1])))
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

    g.emit("add x12, x12, #32")
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
    gbase points at the first row of the group (8 columns, 128 bytes/row).
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
        E("ldp q0, q1, [x0, #%d]" % (32 * lg))  # row r: re, im
        E("ldp q2, q3, [x9, #%d]" % (32 * lg))  # row r+1
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
        E("stp q0, q1, [x0, #%d]" % (32 * lg))
        E("stp q2, q3, [x9, #%d]" % (32 * lg))
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
            E("ldp q%d, q%d, [%s, #%d]" % (a, b, P["rows"][0], 32 * lg))
            E("ldp q%d, q%d, [%s, #%d]" % (c, d, P["rows"][1], 32 * lg))
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
            E("stp q%d, q%d, [%s, #%d]" % (a, b, P["rows"][0], 32 * lg))
            E("stp q%d, q%d, [%s, #%d]" % (c, d, P["rows"][1], 32 * lg))
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
    Transpose between two rows (lane-groups [re c,c+1][im c,c+1]) and the
    scratch format (element q = [re_p, re_pp][im_p, im_pp])."""
    E = g.emit
    g.out.append("")
    g.out.append(".globl %srows_in" % PREFIX)
    g.out.append(".p2align 5")
    g.label(PREFIX + "rows_in")
    E("lsr x3, x3, #2")
    g.label("1")
    for h in range(2):
        E("ldp q0, q1, [x0], #32")
        E("ldp q2, q3, [x1], #32")
        E("trn1 v4.2d, v0.2d, v2.2d")
        E("trn1 v5.2d, v1.2d, v3.2d")
        E("trn2 v6.2d, v0.2d, v2.2d")
        E("trn2 v7.2d, v1.2d, v3.2d")
        E("stp q4, q5, [x2], #32")
        E("stp q6, q7, [x2], #32")
    E("subs x3, x3, #1")
    E("b.ne 1b")
    E("ret")
    g.out.append("")
    g.out.append(".globl %srows_out" % PREFIX)
    g.out.append(".p2align 5")
    g.label(PREFIX + "rows_out")
    E("lsr x3, x3, #2")
    g.label("1")
    for h in range(2):
        E("ldp q4, q5, [x2], #32")
        E("ldp q6, q7, [x2], #32")
        E("trn1 v0.2d, v4.2d, v6.2d")
        E("trn1 v1.2d, v5.2d, v7.2d")
        E("trn2 v2.2d, v4.2d, v6.2d")
        E("trn2 v3.2d, v5.2d, v7.2d")
        E("stp q0, q1, [x0], #32")
        E("stp q2, q3, [x1], #32")
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


def write(outdir, fname, g):
    lines = [".text"] + g.out
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
                gen_stage(g, p, d, mode)
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
    write(outdir, "pass2.S", g)


if __name__ == "__main__":
    main()
