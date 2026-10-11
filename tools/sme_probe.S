// Throughput probes: NEON FMLA, streaming-SVE FMLA (FP64) and SME FMOPA (FP64).
// Each function runs `x0` iterations of 8 independent operations.
.arch armv9-a
.arch_extension sme
.arch_extension sme-f64f64
.text

.globl _probe_neon_fmla
.p2align 4
_probe_neon_fmla:
	stp d8, d9, [sp, #-16]!
	movi v0.2d, #0
	movi v1.2d, #0
	movi v2.2d, #0
	movi v3.2d, #0
	movi v4.2d, #0
	movi v5.2d, #0
	movi v6.2d, #0
	movi v7.2d, #0
	fmov v8.2d, #1.0
	fmov v9.2d, #1.0
1:
	fmla v0.2d, v8.2d, v9.2d
	fmla v1.2d, v8.2d, v9.2d
	fmla v2.2d, v8.2d, v9.2d
	fmla v3.2d, v8.2d, v9.2d
	fmla v4.2d, v8.2d, v9.2d
	fmla v5.2d, v8.2d, v9.2d
	fmla v6.2d, v8.2d, v9.2d
	fmla v7.2d, v8.2d, v9.2d
	fmla v0.2d, v8.2d, v9.2d
	fmla v1.2d, v8.2d, v9.2d
	fmla v2.2d, v8.2d, v9.2d
	fmla v3.2d, v8.2d, v9.2d
	fmla v4.2d, v8.2d, v9.2d
	fmla v5.2d, v8.2d, v9.2d
	fmla v6.2d, v8.2d, v9.2d
	fmla v7.2d, v8.2d, v9.2d
	subs x0, x0, #1
	b.ne 1b
	ldp d8, d9, [sp], #16
	ret

.globl _probe_ssve_fmla
.p2align 4
_probe_ssve_fmla:
	stp d8, d9, [sp, #-64]!
	stp d10, d11, [sp, #16]
	stp d12, d13, [sp, #32]
	stp d14, d15, [sp, #48]
	smstart sm
	ptrue p0.d
	fdup z8.d, #1.0
	fdup z9.d, #1.0
	mov z0.d, #0
	mov z1.d, #0
	mov z2.d, #0
	mov z3.d, #0
	mov z4.d, #0
	mov z5.d, #0
	mov z6.d, #0
	mov z7.d, #0
1:
	fmla z0.d, p0/m, z8.d, z9.d
	fmla z1.d, p0/m, z8.d, z9.d
	fmla z2.d, p0/m, z8.d, z9.d
	fmla z3.d, p0/m, z8.d, z9.d
	fmla z4.d, p0/m, z8.d, z9.d
	fmla z5.d, p0/m, z8.d, z9.d
	fmla z6.d, p0/m, z8.d, z9.d
	fmla z7.d, p0/m, z8.d, z9.d
	subs x0, x0, #1
	b.ne 1b
	smstop sm
	ldp d10, d11, [sp, #16]
	ldp d12, d13, [sp, #32]
	ldp d14, d15, [sp, #48]
	ldp d8, d9, [sp], #64
	ret

.globl _probe_sme_fmopa
.p2align 4
_probe_sme_fmopa:
	stp d8, d9, [sp, #-64]!
	stp d10, d11, [sp, #16]
	stp d12, d13, [sp, #32]
	stp d14, d15, [sp, #48]
	smstart
	ptrue p0.d
	fdup z0.d, #1.0
	fdup z1.d, #1.0
	zero {za}
1:
	fmopa za0.d, p0/m, p0/m, z0.d, z1.d
	fmopa za1.d, p0/m, p0/m, z0.d, z1.d
	fmopa za2.d, p0/m, p0/m, z0.d, z1.d
	fmopa za3.d, p0/m, p0/m, z0.d, z1.d
	fmopa za4.d, p0/m, p0/m, z0.d, z1.d
	fmopa za5.d, p0/m, p0/m, z0.d, z1.d
	fmopa za6.d, p0/m, p0/m, z0.d, z1.d
	fmopa za7.d, p0/m, p0/m, z0.d, z1.d
	subs x0, x0, #1
	b.ne 1b
	smstop
	ldp d10, d11, [sp, #16]
	ldp d12, d13, [sp, #32]
	ldp d14, d15, [sp, #48]
	ldp d8, d9, [sp], #64
	ret

.globl _probe_svl_bytes
.p2align 4
_probe_svl_bytes:
	rdsvl x0, #1
	ret

// Streaming-mode load+store bandwidth: x0 = buffer, x1 = bytes (multiple of 512), x2 = reps
.globl _probe_ssve_copy
.p2align 4
_probe_ssve_copy:
	stp d8, d9, [sp, #-64]!
	stp d10, d11, [sp, #16]
	stp d12, d13, [sp, #32]
	stp d14, d15, [sp, #48]
	smstart sm
	ptrue p0.d
1:
	mov x3, x0
	mov x4, x1
2:
	ld1d {z0.d}, p0/z, [x3]
	ld1d {z1.d}, p0/z, [x3, #1, mul vl]
	ld1d {z2.d}, p0/z, [x3, #2, mul vl]
	ld1d {z3.d}, p0/z, [x3, #3, mul vl]
	ld1d {z4.d}, p0/z, [x3, #4, mul vl]
	ld1d {z5.d}, p0/z, [x3, #5, mul vl]
	ld1d {z6.d}, p0/z, [x3, #6, mul vl]
	ld1d {z7.d}, p0/z, [x3, #7, mul vl]
	st1d {z0.d}, p0, [x3]
	st1d {z1.d}, p0, [x3, #1, mul vl]
	st1d {z2.d}, p0, [x3, #2, mul vl]
	st1d {z3.d}, p0, [x3, #3, mul vl]
	st1d {z4.d}, p0, [x3, #4, mul vl]
	st1d {z5.d}, p0, [x3, #5, mul vl]
	st1d {z6.d}, p0, [x3, #6, mul vl]
	st1d {z7.d}, p0, [x3, #7, mul vl]
	add x3, x3, #512
	subs x4, x4, #512
	b.ne 2b
	subs x2, x2, #1
	b.ne 1b
	smstop sm
	ldp d10, d11, [sp, #16]
	ldp d12, d13, [sp, #32]
	ldp d14, d15, [sp, #48]
	ldp d8, d9, [sp], #64
	ret

// NEON load+store bandwidth for comparison
.globl _probe_neon_copy
.p2align 4
_probe_neon_copy:
1:
	mov x3, x0
	mov x4, x1
2:
	ldp q0, q1, [x3]
	ldp q2, q3, [x3, #32]
	ldp q4, q5, [x3, #64]
	ldp q6, q7, [x3, #96]
	stp q0, q1, [x3]
	stp q2, q3, [x3, #32]
	stp q4, q5, [x3, #64]
	stp q6, q7, [x3, #96]
	add x3, x3, #128
	subs x4, x4, #128
	b.ne 2b
	subs x2, x2, #1
	b.ne 1b
	ret
